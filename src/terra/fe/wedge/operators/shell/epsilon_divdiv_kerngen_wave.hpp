// Wave-parallel Dirichlet/Neumann kernel for EpsilonDivDivKerngen.
//
// This file is included from inside the class body of
// `terra::fe::wedge::operators::shell::EpsilonDivDivKerngen<ScalarT, VecDim>`,
// so the declarations below are class members.
//
// It exposes:
//   - kWave* constants describing the wave layout
//   - team_shmem_size_dn_wave() helper for scratch sizing
//   - run_team_fast_dirichlet_neumann_wave<Diagonal>() team entry point
//
// See epsilon_divdiv_kerngen_wave.md for the full design notes.

#pragma once

// Wave-parallel DN path: each wave processes a radial pencil of 5 cells, with
// both wedges of each cell handled in parallel. Layout: team_size=10 threads
// (one per cell×wedge pair, encoded as team_rank = cell*2 + w) × vector_length=6
// vector lanes (one per wedge node) = 60 of the 64 wavefront lanes used, 4 idle.
// The 6-lane per-wedge reduction is done via Kokkos::parallel_reduce.
static constexpr int kWaveCellsPerWave   = 5;
static constexpr int kWaveWedgesPerCell  = 2;
static constexpr int kWaveLanesPerCell   = 6;
static constexpr int kWaveActiveNodes    = 6;
static constexpr int kWaveThreadsPerWave = kWaveCellsPerWave * kWaveWedgesPerCell; // 10

KOKKOS_INLINE_FUNCTION
size_t team_shmem_size_dn_wave() const
{
    constexpr int nxy  = 4;                     // 2×2 lateral corners
    constexpr int nlev = kWaveCellsPerWave + 1; // 6 radial layers (5 cells need r_0..r_5)
    // coords_sh(nxy,3) + src_sh(nxy,3,nlev) + k_sh(nxy,nlev) + r_sh(nlev)
    const size_t nscalars = size_t( nxy ) * 3 + size_t( nxy ) * 3 * nlev + size_t( nxy ) * nlev + size_t( nlev );
    return sizeof( ScalarType ) * nscalars;
}

/**
 * @brief Team entry for wave-parallel Dirichlet/Neumann matrix-free path.
 *
 * Option B layout: each team is one wave (64 hardware lanes, 60 active). Wave
 * processes a 5-cell radial pencil at fixed (x_cell, y_cell). The 10 team
 * threads encode (cell_in_wave, w) = team_rank / 2, team_rank % 2 — so both
 * wedges of a cell run in parallel.
 *
 * Each thread does ONE wedge of ONE cell. 6-lane reductions (one for k_eval,
 * six for the strain accumulators) use Kokkos::parallel_reduce.
 */
template < bool Diagonal >
KOKKOS_INLINE_FUNCTION void run_team_fast_dirichlet_neumann_wave( const Team& team ) const
{
    // ---------- decode league_rank ----------
    const int r_stacks           = ( hex_rad_ + kWaveCellsPerWave - 1 ) / kWaveCellsPerWave;
    const int per_subdom         = hex_lat_ * hex_lat_ * r_stacks;
    int       rk                 = team.league_rank();
    const int local_subdomain_id = rk / per_subdom;
    rk                           = rk % per_subdom;
    const int r_stack            = rk % r_stacks;
    rk                           = rk / r_stacks;
    const int y_cell             = rk % hex_lat_;
    const int x_cell             = rk / hex_lat_;
    const int r0                 = r_stack * kWaveCellsPerWave;

    // ---------- LDS layout ----------
    constexpr int NXY  = 4;                     // 2x2 lateral corners
    constexpr int NLEV = kWaveCellsPerWave + 1; // 6 radial layers (5 cells)

    double* shmem = reinterpret_cast< double* >( team.team_shmem().get_shmem( team_shmem_size_dn_wave() ) );

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

    // ---------- cooperative LDS load (TeamVectorRange across all 60 active lanes) ----------
    Kokkos::parallel_for( Kokkos::TeamVectorRange( team, NXY ), [&]( int n ) {
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

    Kokkos::parallel_for( Kokkos::TeamVectorRange( team, NLEV ), [&]( int lvl ) {
        const int rr = r0 + lvl;
        r_sh( lvl )  = ( rr <= hex_rad_ ) ? radii_( local_subdomain_id, rr ) : 0.0;
    } );

    constexpr int TOTAL_PAIRS = NXY * NLEV; // 24
    Kokkos::parallel_for( Kokkos::TeamVectorRange( team, TOTAL_PAIRS ), [&]( int t ) {
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
    constexpr double ONE_SIXTH      = 1.0 / 6.0;
    constexpr double NEG_TWO_THIRDS = -0.66666666666666663;

    constexpr double dN_ref[6][3] = {
        { -0.5, -0.5, -ONE_SIXTH },
        { 0.5, 0.0, -ONE_SIXTH },
        { 0.0, 0.5, -ONE_SIXTH },
        { -0.5, -0.5, ONE_SIXTH },
        { 0.5, 0.0, ONE_SIXTH },
        { 0.0, 0.5, ONE_SIXTH } };

    constexpr int WEDGE_NODE_OFF[2][6][3] = {
        { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
        { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

    // ---------- per-(cell, wedge) compute: 10 threads ----------
    Kokkos::parallel_for( Kokkos::TeamThreadRange( team, kWaveThreadsPerWave ), [&]( int cwid ) {
        const int cell_in_wave = cwid / kWaveWedgesPerCell; // 0..4
        const int w            = cwid % kWaveWedgesPerCell; // 0 or 1

        const int  r_cell     = r0 + cell_in_wave;
        const bool cell_valid = ( r_cell < hex_rad_ ) && ( x_cell < hex_lat_ ) && ( y_cell < hex_lat_ );

        const double r_0 = r_sh( cell_in_wave );
        const double r_1 = r_sh( cell_in_wave + 1 );

        const bool at_cmb     = cell_valid && has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
        const bool at_surface = cell_valid && has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );
        const bool at_boundary              = at_cmb || at_surface;
        bool       treat_boundary_dirichlet = false;
        if ( at_boundary )
        {
            const ShellBoundaryFlag sbf = at_cmb ? CMB : SURFACE;
            treat_boundary_dirichlet    = ( get_boundary_condition_flag( bcs_, sbf ) == DIRICHLET );
        }
        const int cmb_shift =
            ( ( at_boundary && treat_boundary_dirichlet && ( !Diagonal ) && at_cmb ) ? 3 : 0 );
        const int surface_shift =
            ( ( at_boundary && treat_boundary_dirichlet && ( !Diagonal ) && at_surface ) ? 3 : 0 );

        // Cell-uniform Jacobian (per-thread; same across this thread's 6 vector lanes).
        const int v0 = ( w == 0 ) ? 0 : 3;
        const int v1 = ( w == 0 ) ? 1 : 2;
        const int v2 = ( w == 0 ) ? 2 : 1;

        const double half_dr = 0.5 * ( r_1 - r_0 );
        const double r_mid   = 0.5 * ( r_0 + r_1 );

        const double J_0_0 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v1, 0 ) );
        const double J_0_1 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v2, 0 ) );
        const double J_0_2 =
            half_dr * ( ONE_THIRD * ( coords_sh( v0, 0 ) + coords_sh( v1, 0 ) + coords_sh( v2, 0 ) ) );
        const double J_1_0 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v1, 1 ) );
        const double J_1_1 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v2, 1 ) );
        const double J_1_2 =
            half_dr * ( ONE_THIRD * ( coords_sh( v0, 1 ) + coords_sh( v1, 1 ) + coords_sh( v2, 1 ) ) );
        const double J_2_0 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v1, 2 ) );
        const double J_2_1 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v2, 2 ) );
        const double J_2_2 =
            half_dr * ( ONE_THIRD * ( coords_sh( v0, 2 ) + coords_sh( v1, 2 ) + coords_sh( v2, 2 ) ) );

        const double J_det = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1 - J_0_1 * J_1_0 * J_2_2 +
                             J_0_1 * J_1_2 * J_2_0 + J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;
        const double inv_det = 1.0 / J_det;

        const double i00 = inv_det * ( J_1_1 * J_2_2 - J_1_2 * J_2_1 );
        const double i01 = inv_det * ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 );
        const double i02 = inv_det * ( J_1_0 * J_2_1 - J_1_1 * J_2_0 );
        const double i10 = inv_det * ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 );
        const double i11 = inv_det * ( J_0_0 * J_2_2 - J_0_2 * J_2_0 );
        const double i12 = inv_det * ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 );
        const double i20 = inv_det * ( J_0_1 * J_1_2 - J_0_2 * J_1_1 );
        const double i21 = inv_det * ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 );
        const double i22 = inv_det * ( J_0_0 * J_1_1 - J_0_1 * J_1_0 );

        double k_sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange( team, 6 ),
            [&]( int node_in_cell, double& sum ) {
                if ( cell_valid )
                {
                    const int knid =
                        WEDGE_NODE_OFF[w][node_in_cell][0] + 2 * WEDGE_NODE_OFF[w][node_in_cell][1];
                    const int klvl = cell_in_wave + WEDGE_NODE_OFF[w][node_in_cell][2];
                    sum += k_sh( knid, klvl );
                }
            },
            k_sum );
        const double k_eval = ONE_SIXTH * k_sum;
        const double kwJ    = k_eval * Kokkos::abs( J_det );

        if constexpr ( !Diagonal )
        {
            double gu00 = 0.0, gu10 = 0.0, gu11 = 0.0;
            double gu20 = 0.0, gu21 = 0.0, gu22 = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange( team, 6 ),
                [&]( int node_in_cell,
                     double& a00,
                     double& a10,
                     double& a11,
                     double& a20,
                     double& a21,
                     double& a22 ) {
                    const bool node_in_range =
                        cell_valid && ( node_in_cell >= cmb_shift ) && ( node_in_cell < 6 - surface_shift );
                    if ( node_in_range )
                    {
                        const double gx = dN_ref[node_in_cell][0];
                        const double gy = dN_ref[node_in_cell][1];
                        const double gz = dN_ref[node_in_cell][2];
                        const double g0 = i00 * gx + i01 * gy + i02 * gz;
                        const double g1 = i10 * gx + i11 * gy + i12 * gz;
                        const double g2 = i20 * gx + i21 * gy + i22 * gz;

                        const int snid =
                            WEDGE_NODE_OFF[w][node_in_cell][0] + 2 * WEDGE_NODE_OFF[w][node_in_cell][1];
                        const int    slvl = cell_in_wave + WEDGE_NODE_OFF[w][node_in_cell][2];
                        const double s0   = src_sh( snid, 0, slvl );
                        const double s1   = src_sh( snid, 1, slvl );
                        const double s2   = src_sh( snid, 2, slvl );

                        a00 += g0 * s0;
                        a10 += 0.5 * ( g1 * s0 + g0 * s1 );
                        a11 += g1 * s1;
                        a20 += 0.5 * ( g2 * s0 + g0 * s2 );
                        a21 += 0.5 * ( g2 * s1 + g1 * s2 );
                        a22 += g2 * s2;
                    }
                },
                gu00,
                gu10,
                gu11,
                gu20,
                gu21,
                gu22 );

            const double div_u = gu00 + gu11 + gu22;

            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange( team, 6 ), [&]( int node_in_cell ) {
                    const bool node_in_range = cell_valid && ( node_in_cell >= cmb_shift ) &&
                                               ( node_in_cell < 6 - surface_shift );
                    if ( node_in_range )
                    {
                        const double gx = dN_ref[node_in_cell][0];
                        const double gy = dN_ref[node_in_cell][1];
                        const double gz = dN_ref[node_in_cell][2];
                        const double g0 = i00 * gx + i01 * gy + i02 * gz;
                        const double g1 = i10 * gx + i11 * gy + i12 * gz;
                        const double g2 = i20 * gx + i21 * gy + i22 * gz;

                        const int ddx = WEDGE_NODE_OFF[w][node_in_cell][0];
                        const int ddy = WEDGE_NODE_OFF[w][node_in_cell][1];
                        const int ddr = WEDGE_NODE_OFF[w][node_in_cell][2];
                        Kokkos::atomic_add(
                            &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 0 ),
                            kwJ * ( 2.0 * ( g0 * gu00 + g1 * gu10 + g2 * gu20 ) +
                                    NEG_TWO_THIRDS * g0 * div_u ) );
                        Kokkos::atomic_add(
                            &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 1 ),
                            kwJ * ( 2.0 * ( g0 * gu10 + g1 * gu11 + g2 * gu21 ) +
                                    NEG_TWO_THIRDS * g1 * div_u ) );
                        Kokkos::atomic_add(
                            &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 2 ),
                            kwJ * ( 2.0 * ( g0 * gu20 + g1 * gu21 + g2 * gu22 ) +
                                    NEG_TWO_THIRDS * g2 * div_u ) );
                    }
                } );
        }

        if constexpr ( Diagonal )
        {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange( team, 6 ), [&]( int node_in_cell ) {
                    const bool node_in_range = cell_valid && ( node_in_cell >= surface_shift ) &&
                                               ( node_in_cell < 6 - cmb_shift );
                    if ( node_in_range )
                    {
                        const double gx = dN_ref[node_in_cell][0];
                        const double gy = dN_ref[node_in_cell][1];
                        const double gz = dN_ref[node_in_cell][2];
                        const double g0 = i00 * gx + i01 * gy + i02 * gz;
                        const double g1 = i10 * gx + i11 * gy + i12 * gz;
                        const double g2 = i20 * gx + i21 * gy + i22 * gz;
                        const double gg = g0 * g0 + g1 * g1 + g2 * g2;

                        const int snid =
                            WEDGE_NODE_OFF[w][node_in_cell][0] + 2 * WEDGE_NODE_OFF[w][node_in_cell][1];
                        const int    slvl = cell_in_wave + WEDGE_NODE_OFF[w][node_in_cell][2];
                        const double sv0  = src_sh( snid, 0, slvl );
                        const double sv1  = src_sh( snid, 1, slvl );
                        const double sv2  = src_sh( snid, 2, slvl );

                        const int ddx = WEDGE_NODE_OFF[w][node_in_cell][0];
                        const int ddy = WEDGE_NODE_OFF[w][node_in_cell][1];
                        const int ddr = WEDGE_NODE_OFF[w][node_in_cell][2];
                        Kokkos::atomic_add(
                            &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 0 ),
                            kwJ * sv0 * ( gg + ONE_THIRD * g0 * g0 ) );
                        Kokkos::atomic_add(
                            &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 1 ),
                            kwJ * sv1 * ( gg + ONE_THIRD * g1 * g1 ) );
                        Kokkos::atomic_add(
                            &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 2 ),
                            kwJ * sv2 * ( gg + ONE_THIRD * g2 * g2 ) );
                    }
                } );
        }
        else if ( treat_boundary_dirichlet && at_boundary )
        {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange( team, 6 ), [&]( int node_in_cell ) {
                    const bool node_in_range = cell_valid && ( node_in_cell >= surface_shift ) &&
                                               ( node_in_cell < 6 - cmb_shift );
                    if ( node_in_range )
                    {
                        const double gx = dN_ref[node_in_cell][0];
                        const double gy = dN_ref[node_in_cell][1];
                        const double gz = dN_ref[node_in_cell][2];
                        const double g0 = i00 * gx + i01 * gy + i02 * gz;
                        const double g1 = i10 * gx + i11 * gy + i12 * gz;
                        const double g2 = i20 * gx + i21 * gy + i22 * gz;
                        const double gg = g0 * g0 + g1 * g1 + g2 * g2;

                        const int snid =
                            WEDGE_NODE_OFF[w][node_in_cell][0] + 2 * WEDGE_NODE_OFF[w][node_in_cell][1];
                        const int    slvl = cell_in_wave + WEDGE_NODE_OFF[w][node_in_cell][2];
                        const double sv0  = src_sh( snid, 0, slvl );
                        const double sv1  = src_sh( snid, 1, slvl );
                        const double sv2  = src_sh( snid, 2, slvl );

                        const int ddx = WEDGE_NODE_OFF[w][node_in_cell][0];
                        const int ddy = WEDGE_NODE_OFF[w][node_in_cell][1];
                        const int ddr = WEDGE_NODE_OFF[w][node_in_cell][2];
                        Kokkos::atomic_add(
                            &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 0 ),
                            kwJ * sv0 * ( gg + ONE_THIRD * g0 * g0 ) );
                        Kokkos::atomic_add(
                            &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 1 ),
                            kwJ * sv1 * ( gg + ONE_THIRD * g1 * g1 ) );
                        Kokkos::atomic_add(
                            &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 2 ),
                            kwJ * sv2 * ( gg + ONE_THIRD * g2 * g2 ) );
                    }
                } );
        }
    } ); // end TeamThreadRange over (cell, wedge) pairs
}
