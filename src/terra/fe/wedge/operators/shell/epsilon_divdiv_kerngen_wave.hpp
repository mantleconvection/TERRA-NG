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

// Wave-parallel DN path: each wave processes a radial pencil of 16 cells in 2
// sequential blocks of 8 cells. Lane layout per block: 8 lanes/cell × 8 cells =
// 64 lanes (full wave). Lanes 6 and 7 of each 8-lane cell-group are padding
// (active wedge has 6 nodes) — enables power-of-2 __shfl_xor reductions.
// Doubling cells-per-wave halves the wave-fixed overhead amortization cost.
static constexpr int kWaveCellsPerBlock = 8;
static constexpr int kWaveBlocksPerWave = 2;
static constexpr int kWaveCellsPerWave  = kWaveCellsPerBlock * kWaveBlocksPerWave;   // 16
static constexpr int kWaveLanesPerCell  = 8;
static constexpr int kWaveActiveNodes   = 6;
static constexpr int kWaveAccumsPerCell = 7;   // gu00, gu10, gu11, gu20, gu21, gu22, div_u

KOKKOS_INLINE_FUNCTION
size_t team_shmem_size_dn_wave() const
{
    constexpr int nxy  = 4;                          // 2×2 lateral corners
    constexpr int nlev = kWaveCellsPerWave + 1;      // 17 radial layers (16 cells need r_0..r_16)
    // coords_sh(nxy,3) + src_sh(nxy,3,nlev) + k_sh(nxy,nlev) + r_sh(nlev)
    const size_t nscalars =
        size_t( nxy ) * 3		//coords_sh
        + size_t( nxy ) * 3 * nlev
        + size_t( nxy ) * nlev
        + size_t( nlev );
    return sizeof( ScalarType ) * nscalars;
}

/**
 * @brief Team entry for wave-parallel Dirichlet/Neumann matrix-free path.
 *
 * Each team is one wave (64 lanes). The wave processes a 16-cell radial
 * pencil at fixed (x_cell, y_cell), in 2 sequential 8-cell blocks. Lane
 * layout per block: 8 lanes/cell × 8 cells = 64 lanes. Lanes 6/7 of each
 * 8-lane cell-group are padding (active wedge has 6 nodes) — enables
 * power-of-2 __shfl_xor reductions across the 8-lane sub-groups.
 *
 * Wedges are handled sequentially inside each cell's 8-lane sub-group.
 * Sub-wave reductions of the 6 per-wedge gradient accumulators use
 * `__shfl_xor` on 8-lane sub-groups; div_u is recovered from the diagonal
 * gradient sum (gu00+gu11+gu22). See epsilon_divdiv_kerngen_wave.md for
 * the full design.
 */
template < bool Diagonal >
KOKKOS_INLINE_FUNCTION void run_team_fast_dirichlet_neumann_wave( const Team& team ) const
{
    // ---------- decode league_rank ----------
    const int r_stacks   = ( hex_rad_ + kWaveCellsPerWave - 1 ) / kWaveCellsPerWave;
    const int per_subdom = hex_lat_ * hex_lat_ * r_stacks;
    int       rk         = team.league_rank();
    const int local_subdomain_id = rk / per_subdom;
    rk                           = rk % per_subdom;
    const int r_stack            = rk % r_stacks;
    rk                           = rk / r_stacks;
    const int y_cell             = rk % hex_lat_;
    const int x_cell             = rk / hex_lat_;
    const int r0                 = r_stack * kWaveCellsPerWave;

    // ---------- LDS layout ----------
    constexpr int NXY  = 4;                              // 2x2 lateral corners
    constexpr int NLEV = kWaveCellsPerWave + 1;          // 17 radial layers (16 cells)

    double* shmem = reinterpret_cast< double* >( team.team_shmem().get_shmem( team_shmem_size_dn_wave() ) );

    using ScratchCoords =
        Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
    using ScratchSrc = Kokkos::
        View< double***, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
    using ScratchK =
        Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
    using ScratchR =
        Kokkos::View< double*, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;

    //ScratchCoords coords_sh( shmem, NXY, 3 );             shmem += NXY * 4;
    //ScratchSrc    src_sh   ( shmem, NXY, 3, NLEV );       shmem += NXY * 3 * NLEV;
    ScratchCoords coords_sh( shmem, 3, NXY );             shmem += NXY * 3;		//Modified dimension 
    ScratchSrc    src_sh   ( shmem, 3, NXY, NLEV );       shmem += NXY * 3 * NLEV;  	//Modified dimensions.
    ScratchK      k_sh     ( shmem, NXY, NLEV );          shmem += NXY * NLEV;
    ScratchR      r_sh     ( shmem, NLEV );               shmem += NLEV;

    // src + k loads: NXY*NLEV = 68 work items spread across 64 lanes.
    constexpr int TOTAL_PAIRS = NXY * NLEV;
    Kokkos::parallel_for( Kokkos::ThreadVectorRange( team, TOTAL_PAIRS ), [&]( int t ) {
        const int node = t / NLEV;  	// (0...NXY)
        const int lvl  = t % NLEV;	// (0...NLEV)
        const int rr   = r0 + lvl;

        if(rr <= hex_rad_)
	{
		if(node == 0)
		{
        		r_sh( lvl )  = radii_( local_subdomain_id, rr );
        	};

        	const int dxn  = node % 2, dyn = node / 2;
        	const int xi   = x_cell + dxn;
        	const int yi   = y_cell + dyn;
        	if ( xi <= hex_lat_ && yi <= hex_lat_)
        	{
            		k_sh( node, lvl )      = k_( local_subdomain_id, xi, yi, rr );
            		src_sh( 0, node, lvl ) = src_( local_subdomain_id, xi, yi, rr, 0 );
            		src_sh( 1, node, lvl ) = src_( local_subdomain_id, xi, yi, rr, 1 );
            		src_sh( 2, node, lvl ) = src_( local_subdomain_id, xi, yi, rr, 2 );

            		if((node % 4) == 0)
			{
            			coords_sh(0, node) = grid_( local_subdomain_id, xi, yi, 0 );
            			coords_sh(1, node) = grid_( local_subdomain_id, xi, yi, 1 );
            			coords_sh(2, node) = grid_( local_subdomain_id, xi, yi, 2 );
            		};
            	}
            	else
            	{
            		k_sh( node, lvl )      = 0.0;

            		src_sh( 0, node, lvl ) = 0.0;
            		src_sh( 1, node, lvl ) = 0.0;
            		src_sh( 2, node, lvl ) = 0.0;

            		if((node % 4) == 0)
			{
            			coords_sh(0, node) = 0.0;
            			coords_sh(1, node) = 0.0;
            			coords_sh(2, node) = 0.0;
            		};
            	}
        }
        else
        {
        	if(node == 0)
		{
			r_sh(lvl) = 0.0;
		};

		if((node % 4) == 0)
		{
            		k_sh( node, lvl )      = 0.0;
            		src_sh( 0, node, lvl ) = 0.0;
            		src_sh( 1, node, lvl ) = 0.0;
            		src_sh( 2, node, lvl ) = 0.0;

            		//This currently writes multiple times to the same location.
            		coords_sh(0, node) = 0.0;
            		coords_sh(1, node) = 0.0;
            		coords_sh(2, node) = 0.0;
            	};
        }
    } );

    team.team_barrier();

    // ---------- shared constants ----------
    constexpr double ONE_THIRD      = 1.0 / 3.0;
    constexpr double ONE_SIXTH      = 1.0 / 6.0;
    constexpr double NEG_TWO_THIRDS = -0.66666666666666663;

    constexpr double dN_ref[6][3] = {
        { -0.5, -0.5, -ONE_SIXTH },
        {  0.5,  0.0, -ONE_SIXTH },
        {  0.0,  0.5, -ONE_SIXTH },
        { -0.5, -0.5,  ONE_SIXTH },
        {  0.5,  0.0,  ONE_SIXTH },
        {  0.0,  0.5,  ONE_SIXTH } };

    constexpr int WEDGE_NODE_OFF[2][6][3] = {
        { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
        { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

    // Lateral corner ids in coords_sh (2x2 grid, row-major: idx = dx + 2*dy).
    // n00 = corner (0,0) = 0, n10 = (1,0) = 1, n01 = (0,1) = 2, n11 = (1,1) = 3.

    // Reduce a value across an 8-lane sub-group via three __shfl_xor rounds.
    // After this, every lane in the group holds the sum of its group's 8 lane values.
    auto wave_reduce8 = [] ( double v ) {
        v += __shfl_xor( v, 1, 8 );
        v += __shfl_xor( v, 2, 8 );
        v += __shfl_xor( v, 4, 8 );
        return v;
    };

    // ---------- wave-parallel per-cell compute (2 sequential blocks × 8 cells) ----------
    Kokkos::parallel_for( Kokkos::ThreadVectorRange( team, 64 ), [&]( int lane_id ) {
        const int cell_in_block = lane_id / kWaveLanesPerCell;   // 0..7
        const int node_in_cell  = lane_id % kWaveLanesPerCell;   // 0..7 (only 0..5 active)
        const bool node_active  = ( node_in_cell < kWaveActiveNodes );

      for ( int block = 0; block < kWaveBlocksPerWave; ++block )
      {
        const int cell_in_wave = block * kWaveCellsPerBlock + cell_in_block;   // 0..15
        const int r_cell       = r0 + cell_in_wave;
        const bool cell_valid  = ( r_cell < hex_rad_ ) && ( x_cell < hex_lat_ ) && ( y_cell < hex_lat_ );
        const bool active      = cell_valid && node_active;

        // r_0/r_1 are shared by the cell's 8 lanes (same cell_in_wave).
        const double r_0 = r_sh( cell_in_wave );
        const double r_1 = r_sh( cell_in_wave + 1 );

        // Boundary state (uniform across the cell's 8 lanes).
        bool at_cmb_tmp, at_surface_tmp, at_boundary_tmp;
        if(cell_valid)
	{
		//Add a special function that handles the NO_FLAG case.
        	at_cmb_tmp     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell,     CMB );
        	at_surface_tmp = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );
	}
	else
	{
		at_cmb_tmp = false;
		at_surface_tmp = false;
	}
	const bool at_cmb = at_cmb_tmp;
	const bool at_surface = at_surface_tmp;
	const bool at_boundary = at_cmb_tmp || at_surface_tmp;

        bool treat_boundary_dirichlet = false;
        if ( at_boundary )
        {
            const ShellBoundaryFlag sbf = at_cmb ? CMB : SURFACE;
            treat_boundary_dirichlet    = ( get_boundary_condition_flag( bcs_, sbf ) == DIRICHLET );
        }
        const int cmb_shift =
            ( ( at_boundary && treat_boundary_dirichlet && ( !Diagonal ) && at_cmb )     ? 3 : 0 );
        const int surface_shift =
            ( ( at_boundary && treat_boundary_dirichlet && ( !Diagonal ) && at_surface ) ? 3 : 0 );

        for ( int w = 0; w < 2; ++w )
        {
            // Compute Jacobian, det, kwJ, and the per-lane physical gradient
            // (g0,g1,g2) in a tight scope so the 9 Jacobian entries + reciprocal
            // + helpers go out of scope before Phase 1/2.
            double kwJ;
            double g0, g1, g2;
            {
                // Cell-wide vertex selection (same on all 8 lanes of the cell).
                const int v0 = ( w == 0 ) ? 0 : 3;   // n00 or n11
                const int v1 = ( w == 0 ) ? 1 : 2;   // n10 or n01
                const int v2 = ( w == 0 ) ? 2 : 1;   // n01 or n10

                const double half_dr = 0.5 * ( r_1 - r_0 );
                const double r_mid   = 0.5 * ( r_0 + r_1 );

                //Loading it only once might be good.
                const double J_0_0 = r_mid * ( -coords_sh( 0, v0 ) + coords_sh( 0, v1 ) );
                const double J_0_1 = r_mid * ( -coords_sh( 0, v0 ) + coords_sh( 0, v2 ) );
                const double J_0_2 =
                    half_dr * ( ONE_THIRD * ( coords_sh( 0, v0 ) + coords_sh( 0, v1 ) + coords_sh( 0, v2 ) ) );
                const double J_1_0 = r_mid * ( -coords_sh( 1, v0 ) + coords_sh( 1, v1 ) );
                const double J_1_1 = r_mid * ( -coords_sh( 1, v0 ) + coords_sh( 1, v2 ) );
                const double J_1_2 =
                    half_dr * ( ONE_THIRD * ( coords_sh( 1, v0 ) + coords_sh( 1, v1 ) + coords_sh( 1, v2 ) ) );
                const double J_2_0 = r_mid * ( -coords_sh( 2, v0 ) + coords_sh( 2, v1 ) );
                const double J_2_1 = r_mid * ( -coords_sh( 2, v0 ) + coords_sh( 2, v2 ) );
                const double J_2_2 =
                    half_dr * ( ONE_THIRD * ( coords_sh( 2, v0 ) + coords_sh( 2, v1 ) + coords_sh( 2, v2) ) );

                const double J_det = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1 - J_0_1 * J_1_0 * J_2_2 +
                                     J_0_1 * J_1_2 * J_2_0 + J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;

                double my_k = 0.0;
                if ( active )
                {
                    const int knid = WEDGE_NODE_OFF[w][node_in_cell][0]
                                   + 2 * WEDGE_NODE_OFF[w][node_in_cell][1];
                    const int klvl = cell_in_wave + WEDGE_NODE_OFF[w][node_in_cell][2];
                    my_k = k_sh( knid, klvl );
                }
                const double k_eval = ONE_SIXTH * wave_reduce8( my_k );
                kwJ = k_eval * J_det;

                const double inv_det = 1.0 / J_det;
                const double gx = dN_ref[node_in_cell][0];
                const double gy = dN_ref[node_in_cell][1];
                const double gz = dN_ref[node_in_cell][2];
                g0 = inv_det * (
                    (  J_1_1 * J_2_2 - J_1_2 * J_2_1 ) * gx +
                    ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 ) * gy +
                    (  J_1_0 * J_2_1 - J_1_1 * J_2_0 ) * gz );
                g1 = inv_det * (
                    ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 ) * gx +
                    (  J_0_0 * J_2_2 - J_0_2 * J_2_0 ) * gy +
                    ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 ) * gz );
                g2 = inv_det * (
                    (  J_0_1 * J_1_2 - J_0_2 * J_1_1 ) * gx +
                    ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 ) * gy +
                    (  J_0_0 * J_1_1 - J_0_1 * J_1_0 ) * gz );
            } // J_*_*, J_det, inv_det, gx/gy/gz, my_k, k_eval dead

            // ===== Phase 1: per-lane gather + wave-reduce (matvec only) =====
            if constexpr ( !Diagonal )
            {
                const bool node_in_range = active
                                        && ( node_in_cell >= cmb_shift )
                                        && ( node_in_cell <  6 - surface_shift );

                double p_gu00 = 0.0, p_gu10 = 0.0, p_gu11 = 0.0;
                double p_gu20 = 0.0, p_gu21 = 0.0, p_gu22 = 0.0;
                if ( node_in_range )
                {
                    const int    snid = WEDGE_NODE_OFF[w][node_in_cell][0]
                                      + 2 * WEDGE_NODE_OFF[w][node_in_cell][1];
                    const int    slvl = cell_in_wave + WEDGE_NODE_OFF[w][node_in_cell][2];
                    const double s0   = src_sh( 0, snid, slvl );
                    const double s1   = src_sh( 1, snid, slvl );
                    const double s2   = src_sh( 2, snid, slvl );

                    p_gu00 = g0 * s0;
                    p_gu10 = 0.5 * ( g1 * s0 + g0 * s1 );
                    p_gu11 = g1 * s1;
                    p_gu20 = 0.5 * ( g2 * s0 + g0 * s2 );
                    p_gu21 = 0.5 * ( g2 * s1 + g1 * s2 );
                    p_gu22 = g2 * s2;
                }

                // Wave-reduce 6 accumulators over each 8-lane cell-group.
                const double gu00  = wave_reduce8( p_gu00 );
                const double gu10  = wave_reduce8( p_gu10 );
                const double gu11  = wave_reduce8( p_gu11 );
                const double gu20  = wave_reduce8( p_gu20 );
                const double gu21  = wave_reduce8( p_gu21 );
                const double gu22  = wave_reduce8( p_gu22 );
                const double div_u = gu00 + gu11 + gu22;

                // ===== Phase 2 (matvec branch): per-lane scatter =====
                if ( node_in_range )
                {
                    const int ddx = WEDGE_NODE_OFF[w][node_in_cell][0];
                    const int ddy = WEDGE_NODE_OFF[w][node_in_cell][1];
                    const int ddr = WEDGE_NODE_OFF[w][node_in_cell][2];
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 0 ),
                        kwJ * ( 2.0 * ( g0 * gu00 + g1 * gu10 + g2 * gu20 ) + NEG_TWO_THIRDS * g0 * div_u ) );
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 1 ),
                        kwJ * ( 2.0 * ( g0 * gu10 + g1 * gu11 + g2 * gu21 ) + NEG_TWO_THIRDS * g1 * div_u ) );
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 2 ),
                        kwJ * ( 2.0 * ( g0 * gu20 + g1 * gu21 + g2 * gu22 ) + NEG_TWO_THIRDS * g2 * div_u ) );
                }
            }

            // ===== Phase 2 (diagonal or boundary-dirichlet branch) =====
            if constexpr ( Diagonal )
            {
                const bool node_in_range = active
                                        && ( node_in_cell >= surface_shift )
                                        && ( node_in_cell <  6 - cmb_shift );
                if ( node_in_range )
                {
                    const double gg = g0 * g0 + g1 * g1 + g2 * g2;

                    const int    snid = WEDGE_NODE_OFF[w][node_in_cell][0]
                                      + 2 * WEDGE_NODE_OFF[w][node_in_cell][1];
                    const int    slvl = cell_in_wave + WEDGE_NODE_OFF[w][node_in_cell][2];
                    const double sv0  = src_sh( 0, snid, slvl );
                    const double sv1  = src_sh( 1, snid, slvl );
                    const double sv2  = src_sh( 2, snid, slvl );

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
            }
            else if ( treat_boundary_dirichlet && at_boundary )
            {
                // Mixed case for matvec at a Dirichlet boundary.
                const bool node_in_range = active
                                        && ( node_in_cell >= surface_shift )
                                        && ( node_in_cell <  6 - cmb_shift );
                if ( node_in_range )
                {
                    const double gg = g0 * g0 + g1 * g1 + g2 * g2;

                    const int    snid = WEDGE_NODE_OFF[w][node_in_cell][0]
                                      + 2 * WEDGE_NODE_OFF[w][node_in_cell][1];
                    const int    slvl = cell_in_wave + WEDGE_NODE_OFF[w][node_in_cell][2];
                    const double sv0  = src_sh( 0, snid, slvl );
                    const double sv1  = src_sh( 1, snid, slvl );
                    const double sv2  = src_sh( 2, snid, slvl );

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
            }
        } // end wedge loop
      } // end block loop
    } );
}
