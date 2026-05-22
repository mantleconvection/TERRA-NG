// Hex 2x2x2 Gauss Dirichlet/Neumann kernel for EpsilonDivDivKerngen.
//
// This file is included from inside the class body of
// `terra::fe::wedge::operators::shell::EpsilonDivDivKerngen<ScalarT, VecDim>`,
// so the declarations below are class members.
//
// It exposes:
//   - kHex* constants describing the hex-cell wave layout
//   - team_shmem_size_dn_hex() helper for scratch sizing
//   - run_team_fast_dirichlet_neumann_hex<Diagonal>() team entry point
//
// Full 2x2x2 Gauss integration on the trilinear Q1 hex (no wedge subdivision).
// 1-pt Gauss on Q1 hex is rank-deficient (hourglass modes) and unsuitable as a
// standalone matvec kernel for an iterative solver. 2x2x2 is the standard
// minimum "fully integrated" rule that gives a full-rank local stiffness on
// any non-degenerate hex.
//
// See doc/epsdivdiv_benchmarks/wave_and_hex_kernels.md for the full design notes.

#pragma once

// Hex 2x2x2 Gauss DN path: each wave processes a radial pencil of 16 cells in
// 2 sequential blocks of 8 cells. Lane layout per block: 8 lanes/cell × 8
// cells = 64 lanes (full wave). All 8 lanes are active (one per hex corner);
// no padding. Each cell iteration loops over 8 quadrature points internally,
// accumulating matvec/diagonal contributions per lane and scattering once.
static constexpr int kHexCellsPerBlock = 8;
static constexpr int kHexBlocksPerWave = 2;
static constexpr int kHexCellsPerWave  = kHexCellsPerBlock * kHexBlocksPerWave; // 16
static constexpr int kHexLanesPerCell  = 8;
static constexpr int kHexActiveNodes   = 8; // no padding
static constexpr int kHexQuadPoints    = 8; // 2x2x2 Gauss
static constexpr int kHexAccumsPerCell = 6; // gu00, gu10, gu11, gu20, gu21, gu22

KOKKOS_INLINE_FUNCTION
size_t team_shmem_size_dn_hex() const
{
    constexpr int nxy  = 4;                    // 2x2 lateral corners
    constexpr int nlev = kHexCellsPerWave + 1; // 17 radial layers
    // coords_sh(nxy,3) + src_sh(nxy,3,nlev) + k_sh(nxy,nlev) + r_sh(nlev)
    const size_t nscalars = size_t( nxy ) * 3 + size_t( nxy ) * 3 * nlev + size_t( nxy ) * nlev + size_t( nlev );
    return sizeof( ScalarType ) * nscalars;
}

/**
 * @brief Team entry for hex 2x2x2 Gauss Dirichlet/Neumann matrix-free path.
 *
 * Each team is one wave (64 lanes). The wave processes a 16-cell radial pencil
 * at fixed (x_cell, y_cell) in 2 sequential 8-cell blocks. Lane layout per
 * block: 8 lanes/cell × 8 cells = 64 lanes; every lane corresponds to one of
 * the 8 hex corners (no padding).
 *
 * Within each cell iteration, the kernel loops over 8 Gauss-Legendre 2-pt
 * quadrature points (on the [0,1]^3 reference, weight = 1/8 each, points at
 * 1/2 ± 1/(2 sqrt(3))). At each quad point:
 *   - the cell-wide Jacobian J(q) is computed (same on all 8 lanes of the cell)
 *   - each lane evaluates its corner's shape-function gradient at q
 *   - the strain ε(u)(q) is built via 6 wave_reduce8 calls across the 8 lanes
 *   - per lane, the matvec/diagonal contribution at q is added to a local
 *     accumulator
 *
 * One scatter per lane at the end (3 atomic_adds for matvec, +3 more for a
 * boundary-Dirichlet face corner if applicable).
 */
template < bool Diagonal >
KOKKOS_INLINE_FUNCTION void run_team_fast_dirichlet_neumann_hex( const Team& team ) const
{
    // ---------- decode league_rank ----------
    const int r_stacks           = ( hex_rad_ + kHexCellsPerWave - 1 ) / kHexCellsPerWave;
    const int per_subdom         = hex_lat_ * hex_lat_ * r_stacks;
    int       rk                 = team.league_rank();
    const int local_subdomain_id = rk / per_subdom;
    rk                           = rk % per_subdom;
    const int r_stack            = rk % r_stacks;
    rk                           = rk / r_stacks;
    const int y_cell             = rk % hex_lat_;
    const int x_cell             = rk / hex_lat_;
    const int r0                 = r_stack * kHexCellsPerWave;

    // ---------- LDS layout (identical to the wedge wave path) ----------
    constexpr int NXY  = 4;                    // 2x2 lateral corners
    constexpr int NLEV = kHexCellsPerWave + 1; // 17 radial layers (16 cells)

    double* shmem = reinterpret_cast< double* >( team.team_shmem().get_shmem( team_shmem_size_dn_hex() ) );

    using ScratchCoords =
        Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
    using ScratchSrc =
        Kokkos::View< double***, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
    using ScratchK =
        Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
    using ScratchR =
        Kokkos::View< double*, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;

    ScratchCoords coords_sh( shmem, NXY, 3 );
    shmem += NXY * 3;
    ScratchSrc src_sh( shmem, NXY, 3, NLEV );
    shmem += NXY * 3 * NLEV;
    ScratchK k_sh( shmem, NXY, NLEV );
    shmem += NXY * NLEV;
    ScratchR r_sh( shmem, NLEV );
    shmem += NLEV;

    // ---------- cooperative LDS load (64 lanes) ----------
    Kokkos::parallel_for( Kokkos::ThreadVectorRange( team, NXY ), [&]( int n ) {
        const int dxn = n % 2, dyn = n / 2;
        const int xi = x_cell + dxn;
        const int yi = y_cell + dyn;
        if ( xi <= hex_lat_ && yi <= hex_lat_ )
        {
            coords_sh( n, 0 ) = grid_( local_subdomain_id, xi, yi, 0 );
            coords_sh( n, 1 ) = grid_( local_subdomain_id, xi, yi, 1 );
            coords_sh( n, 2 ) = grid_( local_subdomain_id, xi, yi, 2 );
        }
        else
        {
            coords_sh( n, 0 ) = coords_sh( n, 1 ) = coords_sh( n, 2 ) = 0.0;
        }
    } );

    Kokkos::parallel_for( Kokkos::ThreadVectorRange( team, NLEV ), [&]( int lvl ) {
        const int rr = r0 + lvl;
        r_sh( lvl )  = ( rr <= hex_rad_ ) ? radii_( local_subdomain_id, rr ) : 0.0;
    } );

    constexpr int TOTAL_PAIRS = NXY * NLEV;
    Kokkos::parallel_for( Kokkos::ThreadVectorRange( team, TOTAL_PAIRS ), [&]( int t ) {
        const int node = t / NLEV;
        const int lvl  = t % NLEV;
        const int dxn = node % 2, dyn = node / 2;
        const int xi = x_cell + dxn;
        const int yi = y_cell + dyn;
        const int rr = r0 + lvl;
        if ( xi <= hex_lat_ && yi <= hex_lat_ && rr <= hex_rad_ )
        {
            k_sh( node, lvl )      = k_( local_subdomain_id, xi, yi, rr );
            src_sh( node, 0, lvl ) = src_( local_subdomain_id, xi, yi, rr, 0 );
            src_sh( node, 1, lvl ) = src_( local_subdomain_id, xi, yi, rr, 1 );
            src_sh( node, 2, lvl ) = src_( local_subdomain_id, xi, yi, rr, 2 );
        }
        else
        {
            k_sh( node, lvl )      = 0.0;
            src_sh( node, 0, lvl ) = src_sh( node, 1, lvl ) = src_sh( node, 2, lvl ) = 0.0;
        }
    } );

    team.team_barrier();

    // ---------- shared constants ----------
    constexpr double ONE_THIRD      = 1.0 / 3.0;
    constexpr double NEG_TWO_THIRDS = -0.66666666666666663;

    // Gauss-Legendre 2-pt on [0,1]: nodes at 1/2 ± 1/(2 sqrt(3)), each with
    // weight 1/2. The 2x2x2 product rule gives 8 quad points, each with
    // weight (1/2)^3 = 1/8 on the [0,1]^3 reference cube.
    constexpr double GAUSS_A = 0.21132486540518713; // 1/2 - 1/(2 sqrt(3))
    constexpr double GAUSS_B = 0.78867513459481287; // 1/2 + 1/(2 sqrt(3))
    constexpr double GAUSS_W = 0.125;               // 1/8

    // Reduce a value across an 8-lane sub-group via three __shfl_xor rounds.
    auto wave_reduce8 = []( double v ) {
        v += __shfl_xor( v, 1, 8 );
        v += __shfl_xor( v, 2, 8 );
        v += __shfl_xor( v, 4, 8 );
        return v;
    };

    // ---------- wave-parallel per-cell compute (2 sequential blocks × 8 cells) ----------
    Kokkos::parallel_for( Kokkos::ThreadVectorRange( team, 64 ), [&]( int lane_id ) {
        const int    cell_in_block = lane_id / kHexLanesPerCell; // 0..7
        const int    node_in_cell  = lane_id % kHexLanesPerCell; // 0..7 (all hex corners active)
        const int    dxn           = node_in_cell & 1;
        const int    dyn           = ( node_in_cell >> 1 ) & 1;
        const int    dzn           = ( node_in_cell >> 2 ) & 1;
        const double s_x           = ( dxn == 0 ) ? -1.0 : 1.0;
        const double s_y           = ( dyn == 0 ) ? -1.0 : 1.0;
        const double s_z           = ( dzn == 0 ) ? -1.0 : 1.0;

        for ( int block = 0; block < kHexBlocksPerWave; ++block )
        {
            const int  cell_in_wave = block * kHexCellsPerBlock + cell_in_block; // 0..15
            const int  r_cell       = r0 + cell_in_wave;
            const bool cell_valid   = ( r_cell < hex_rad_ ) && ( x_cell < hex_lat_ ) && ( y_cell < hex_lat_ );

            const double r_0 = r_sh( cell_in_wave );
            const double r_1 = r_sh( cell_in_wave + 1 );
            const double dr  = r_1 - r_0;

            // Boundary state (uniform across the cell's 8 lanes).
            const bool at_cmb     = cell_valid && has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
            const bool at_surface = cell_valid && has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );
            const bool at_boundary               = at_cmb || at_surface;
            bool       treat_boundary_dirichlet = false;
            if ( at_boundary )
            {
                const ShellBoundaryFlag sbf = at_cmb ? CMB : SURFACE;
                treat_boundary_dirichlet    = ( get_boundary_condition_flag( bcs_, sbf ) == DIRICHLET );
            }

            // Per-corner face flags: CMB face = dzn=0, SURFACE face = dzn=1.
            const bool on_cmb_face = ( dzn == 0 );
            const bool on_sur_face = ( dzn == 1 );

            // Per-corner LDS reads (this lane's hex corner).
            const int    snid = dxn + 2 * dyn;       // 0..3 lateral corner id
            const int    slvl = cell_in_wave + dzn;  // radial level
            const double s0   = cell_valid ? src_sh( snid, 0, slvl ) : 0.0;
            const double s1   = cell_valid ? src_sh( snid, 1, slvl ) : 0.0;
            const double s2   = cell_valid ? src_sh( snid, 2, slvl ) : 0.0;
            const double my_k = cell_valid ? k_sh( snid, slvl ) : 0.0;

            // In-range flags for matvec normal branch and bdry-Dirichlet branch.
            // The bdry-Dirichlet branch only fires for cells with a Dirichlet
            // boundary — Neumann boundary corners contribute to matvec normally.
            const bool matvec_in_range = cell_valid &&
                                         !( treat_boundary_dirichlet && at_cmb && on_cmb_face ) &&
                                         !( treat_boundary_dirichlet && at_surface && on_sur_face );
            const bool bdry_in_range   = cell_valid && treat_boundary_dirichlet &&
                                       ( ( at_cmb && on_cmb_face ) || ( at_surface && on_sur_face ) );

            // Per-lane accumulators across the 8 quadrature points.
            //   val_*  : matvec contribution at this lane's corner (used in !Diagonal)
            //   diag_* : diagonal-like contribution at this lane's corner
            //            (used by Diagonal mode and by the matvec bdry-Dirichlet branch)
            double val_x  = 0.0, val_y  = 0.0, val_z  = 0.0;
            double diag_x = 0.0, diag_y = 0.0, diag_z = 0.0;

            // ============ 2x2x2 Gauss point loop ============
            for ( int qzi = 0; qzi < 2; ++qzi )
            {
                const double zeta = ( qzi == 0 ) ? GAUSS_A : GAUSS_B;
                const double r_q  = ( 1.0 - zeta ) * r_0 + zeta * r_1;
                // 1D shape function value M_z for THIS lane at this qz:
                //   qzi == dzn → 1D node "matches" → value is the "B" (larger) end
                //   qzi != dzn → 1D node "opposite" → value is "A"
                const double M_z = ( qzi == dzn ) ? GAUSS_B : GAUSS_A;

                for ( int qyi = 0; qyi < 2; ++qyi )
                {
                    const double eta = ( qyi == 0 ) ? GAUSS_A : GAUSS_B;
                    const double M_y = ( qyi == dyn ) ? GAUSS_B : GAUSS_A;

                    // ĉ_xi(η) = (1-η)(c_10 - c_00) + η(c_11 - c_01).
                    // Independent of ξ (and ζ) → hoist out of the inner qx loop.
                    const double cx_xi = ( 1.0 - eta ) * ( coords_sh( 1, 0 ) - coords_sh( 0, 0 ) ) +
                                         eta * ( coords_sh( 3, 0 ) - coords_sh( 2, 0 ) );
                    const double cy_xi = ( 1.0 - eta ) * ( coords_sh( 1, 1 ) - coords_sh( 0, 1 ) ) +
                                         eta * ( coords_sh( 3, 1 ) - coords_sh( 2, 1 ) );
                    const double cz_xi = ( 1.0 - eta ) * ( coords_sh( 1, 2 ) - coords_sh( 0, 2 ) ) +
                                         eta * ( coords_sh( 3, 2 ) - coords_sh( 2, 2 ) );

                    // J(:, ξ) = r(ζ) · ĉ_xi(η) — independent of qx, hoist.
                    const double J_0_0 = r_q * cx_xi;
                    const double J_1_0 = r_q * cy_xi;
                    const double J_2_0 = r_q * cz_xi;

                    for ( int qxi = 0; qxi < 2; ++qxi )
                    {
                        const double xi  = ( qxi == 0 ) ? GAUSS_A : GAUSS_B;
                        const double M_x = ( qxi == dxn ) ? GAUSS_B : GAUSS_A;

                        // ĉ_eta(ξ) = (1-ξ)(c_01 - c_00) + ξ(c_11 - c_10)
                        const double cx_eta = ( 1.0 - xi ) * ( coords_sh( 2, 0 ) - coords_sh( 0, 0 ) ) +
                                              xi * ( coords_sh( 3, 0 ) - coords_sh( 1, 0 ) );
                        const double cy_eta = ( 1.0 - xi ) * ( coords_sh( 2, 1 ) - coords_sh( 0, 1 ) ) +
                                              xi * ( coords_sh( 3, 1 ) - coords_sh( 1, 1 ) );
                        const double cz_eta = ( 1.0 - xi ) * ( coords_sh( 2, 2 ) - coords_sh( 0, 2 ) ) +
                                              xi * ( coords_sh( 3, 2 ) - coords_sh( 1, 2 ) );

                        // ĉ_centroid(ξ, η) = bilinear interp of 4 lateral corners
                        const double xa = 1.0 - xi;
                        const double xb = xi;
                        const double ya = 1.0 - eta;
                        const double yb = eta;
                        const double cx_c = xa * ya * coords_sh( 0, 0 ) + xb * ya * coords_sh( 1, 0 ) +
                                            xa * yb * coords_sh( 2, 0 ) + xb * yb * coords_sh( 3, 0 );
                        const double cy_c = xa * ya * coords_sh( 0, 1 ) + xb * ya * coords_sh( 1, 1 ) +
                                            xa * yb * coords_sh( 2, 1 ) + xb * yb * coords_sh( 3, 1 );
                        const double cz_c = xa * ya * coords_sh( 0, 2 ) + xb * ya * coords_sh( 1, 2 ) +
                                            xa * yb * coords_sh( 2, 2 ) + xb * yb * coords_sh( 3, 2 );

                        // J(:, η) = r(ζ) · ĉ_eta(ξ);  J(:, ζ) = dr · ĉ_centroid(ξ,η).
                        const double J_0_1 = r_q * cx_eta;
                        const double J_1_1 = r_q * cy_eta;
                        const double J_2_1 = r_q * cz_eta;
                        const double J_0_2 = dr * cx_c;
                        const double J_1_2 = dr * cy_c;
                        const double J_2_2 = dr * cz_c;

                        const double J_det = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1 - J_0_1 * J_1_0 * J_2_2 +
                                             J_0_1 * J_1_2 * J_2_0 + J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;
                        const double inv_det = cell_valid ? ( 1.0 / J_det ) : 0.0;
                        const double abs_det = Kokkos::abs( J_det );

                        // Reference-frame shape-function gradient at this quad
                        // point for this lane:
                        //   dN_i/dξ = s_x_i · M_y_i(η_q) · M_z_i(ζ_q)
                        //   dN_i/dη = M_x_i(ξ_q) · s_y_i · M_z_i(ζ_q)
                        //   dN_i/dζ = M_x_i(ξ_q) · M_y_i(η_q) · s_z_i
                        const double gx_ref = s_x * M_y * M_z;
                        const double gy_ref = M_x * s_y * M_z;
                        const double gz_ref = M_x * M_y * s_z;

                        // Physical gradient g_i = inv_J^T · grad_ref(N_i)
                        const double g0 =
                            inv_det * ( ( J_1_1 * J_2_2 - J_1_2 * J_2_1 ) * gx_ref +
                                        ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 ) * gy_ref +
                                        ( J_1_0 * J_2_1 - J_1_1 * J_2_0 ) * gz_ref );
                        const double g1 =
                            inv_det * ( ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 ) * gx_ref +
                                        ( J_0_0 * J_2_2 - J_0_2 * J_2_0 ) * gy_ref +
                                        ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 ) * gz_ref );
                        const double g2 =
                            inv_det * ( ( J_0_1 * J_1_2 - J_0_2 * J_1_1 ) * gx_ref +
                                        ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 ) * gy_ref +
                                        ( J_0_0 * J_1_1 - J_0_1 * J_1_0 ) * gz_ref );

                        // k at this quad point via N_i(q) · k_i interpolation
                        // (8-corner bilinear), realised as a wave_reduce8.
                        const double Ni_q   = M_x * M_y * M_z;
                        const double k_eval = wave_reduce8( Ni_q * my_k );
                        const double kwJ    = GAUSS_W * k_eval * abs_det;

                        // ε(u)(q) — each lane contributes its corner's gradient
                        // ⊗ src; wave_reduce8 sums over the 8 corners.
                        const double p_gu00 = matvec_in_range ? ( g0 * s0 ) : 0.0;
                        const double p_gu10 = matvec_in_range ? ( 0.5 * ( g1 * s0 + g0 * s1 ) ) : 0.0;
                        const double p_gu11 = matvec_in_range ? ( g1 * s1 ) : 0.0;
                        const double p_gu20 = matvec_in_range ? ( 0.5 * ( g2 * s0 + g0 * s2 ) ) : 0.0;
                        const double p_gu21 = matvec_in_range ? ( 0.5 * ( g2 * s1 + g1 * s2 ) ) : 0.0;
                        const double p_gu22 = matvec_in_range ? ( g2 * s2 ) : 0.0;

                        if constexpr ( !Diagonal )
                        {
                            const double gu00  = wave_reduce8( p_gu00 );
                            const double gu10  = wave_reduce8( p_gu10 );
                            const double gu11  = wave_reduce8( p_gu11 );
                            const double gu20  = wave_reduce8( p_gu20 );
                            const double gu21  = wave_reduce8( p_gu21 );
                            const double gu22  = wave_reduce8( p_gu22 );
                            const double div_u = gu00 + gu11 + gu22;

                            if ( matvec_in_range )
                            {
                                val_x += kwJ * ( 2.0 * ( g0 * gu00 + g1 * gu10 + g2 * gu20 ) +
                                                 NEG_TWO_THIRDS * g0 * div_u );
                                val_y += kwJ * ( 2.0 * ( g0 * gu10 + g1 * gu11 + g2 * gu21 ) +
                                                 NEG_TWO_THIRDS * g1 * div_u );
                                val_z += kwJ * ( 2.0 * ( g0 * gu20 + g1 * gu21 + g2 * gu22 ) +
                                                 NEG_TWO_THIRDS * g2 * div_u );
                            }
                            if ( bdry_in_range )
                            {
                                const double gg = g0 * g0 + g1 * g1 + g2 * g2;
                                diag_x += kwJ * s0 * ( gg + ONE_THIRD * g0 * g0 );
                                diag_y += kwJ * s1 * ( gg + ONE_THIRD * g1 * g1 );
                                diag_z += kwJ * s2 * ( gg + ONE_THIRD * g2 * g2 );
                            }
                        }
                        else
                        {
                            if ( cell_valid )
                            {
                                const double gg = g0 * g0 + g1 * g1 + g2 * g2;
                                diag_x += kwJ * s0 * ( gg + ONE_THIRD * g0 * g0 );
                                diag_y += kwJ * s1 * ( gg + ONE_THIRD * g1 * g1 );
                                diag_z += kwJ * s2 * ( gg + ONE_THIRD * g2 * g2 );
                            }
                        }
                    } // qx loop
                } // qy loop
            } // qz loop

            // ============ single scatter per branch after 8 quad points ============
            if constexpr ( !Diagonal )
            {
                if ( matvec_in_range )
                {
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_cell + dxn, y_cell + dyn, r_cell + dzn, 0 ), val_x );
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_cell + dxn, y_cell + dyn, r_cell + dzn, 1 ), val_y );
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_cell + dxn, y_cell + dyn, r_cell + dzn, 2 ), val_z );
                }
                if ( bdry_in_range )
                {
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_cell + dxn, y_cell + dyn, r_cell + dzn, 0 ), diag_x );
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_cell + dxn, y_cell + dyn, r_cell + dzn, 1 ), diag_y );
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_cell + dxn, y_cell + dyn, r_cell + dzn, 2 ), diag_z );
                }
            }
            else
            {
                if ( cell_valid )
                {
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_cell + dxn, y_cell + dyn, r_cell + dzn, 0 ), diag_x );
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_cell + dxn, y_cell + dyn, r_cell + dzn, 1 ), diag_y );
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_cell + dxn, y_cell + dyn, r_cell + dzn, 2 ), diag_z );
                }
            }
        } // end block loop
    } );
}
