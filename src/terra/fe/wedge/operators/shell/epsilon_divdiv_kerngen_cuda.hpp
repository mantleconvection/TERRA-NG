#pragma once
/**
 * EpsilonDivDivKerngenCuda — fast Dirichlet/Neumann matvec, native CUDA.
 *
 * Drop-in wrapper around `EpsilonDivDivKerngen` with the same constructor
 * signature.  Internally extracts raw device pointers from the Kokkos Views
 * and launches a hand-written `__global__` kernel that mirrors the algorithm
 * in `epsilon_divdiv_kerngen.hpp` (operator_fast_dirichlet_neumann_path<false>).
 *
 * Limitations vs the Kokkos version (asserts at construction time):
 *   - matvec only (Diagonal=false)
 *   - fast Dirichlet/Neumann path only (no slow path, no free-slip path)
 *   - no penalty axpy (Stokes-typical Dirichlet/Dirichlet config)
 *   - no MPI communication (single-rank benchmark use)
 *
 * Built only when CUDA is the default execution space.
 */

#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"  // for constructor sig + types
#include "linalg/operator.hpp"
#include "linalg/vector_q1.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "grid/shell/bit_masks.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace terra::fe::wedge::operators::shell {

namespace cuda_kerngen_detail {

// ---------- Tile constants (mirror lat_tile=4, r_tile=8, r_passes=2) ---------
constexpr int LAT_TILE     = 4;
constexpr int R_TILE       = 8;
constexpr int R_PASSES     = 2;
constexpr int R_TILE_BLOCK = R_TILE * R_PASSES;            // 16
constexpr int TEAM_SIZE    = LAT_TILE * LAT_TILE * R_TILE; // 128
constexpr int LB_BLOCKS_PER_SM = 5;

constexpr int NLEV  = R_TILE_BLOCK + 1;
constexpr int N_LAT = LAT_TILE + 1;
constexpr int NXY   = N_LAT * N_LAT;

__host__ __device__ constexpr size_t shmem_bytes_dn()
{
    return sizeof( double ) * ( NXY * 3 + NXY * 3 * NLEV + NXY * NLEV + NLEV );
}

// ShellBoundaryFlag bit pattern matches grid::shell::ShellBoundaryFlag
constexpr uint8_t CMB_BITS     = static_cast< uint8_t >( grid::shell::ShellBoundaryFlag::CMB );
constexpr uint8_t SURFACE_BITS = static_cast< uint8_t >( grid::shell::ShellBoundaryFlag::SURFACE );

__device__ __forceinline__ bool flag_cmb( uint8_t f )     { return ( f & CMB_BITS ) == CMB_BITS; }
__device__ __forceinline__ bool flag_surface( uint8_t f ) { return ( f & SURFACE_BITS ) == SURFACE_BITS; }

struct Extents
{
    int hex_lat, hex_rad, local_subdomains;
    int lat_tiles, r_tiles;
    int nx, ny, nr;
};

struct Views
{
    const double* __restrict__  coords;       // (sub, x, y, 3)        — AoS
    const double* __restrict__  radii;        // (sub, r)
    const double* __restrict__  k;            // (sub, x, y, r)
    const double* __restrict__  src_d[3];     // 3x (sub, x, y, r)     — SoA components
    double*       __restrict__  dst_d[3];     // 3x (sub, x, y, r)     — SoA components
    const uint8_t* __restrict__ bcs;          // (sub, x, y, r)
};

// All read-only loads are routed through __ldg so they hit the read-only
// data cache (RO/L1.RO), matching Kokkos's automatic RandomAccess promotion
// for const Views captured by KOKKOS_CLASS_LAMBDA.
__device__ __forceinline__
double load_coord( const Views& v, const Extents& e, int s, int x, int y, int c )
{ return __ldg( &v.coords[ ( ( (size_t)s * e.nx + x ) * e.ny + y ) * 3 + c ] ); }

__device__ __forceinline__
double load_radius( const Views& v, const Extents& e, int s, int r )
{ return __ldg( &v.radii[ (size_t)s * e.nr + r ] ); }

__device__ __forceinline__
double load_k( const Views& v, const Extents& e, int s, int x, int y, int r )
{ return __ldg( &v.k[ ( ( (size_t)s * e.nx + x ) * e.ny + y ) * e.nr + r ] ); }

__device__ __forceinline__
double load_src( const Views& v, const Extents& e, int s, int x, int y, int r, int d )
{ return __ldg( &v.src_d[d][ ( ( (size_t)s * e.nx + x ) * e.ny + y ) * e.nr + r ] ); }

__device__ __forceinline__
double* dst_ptr( const Views& v, const Extents& e, int s, int x, int y, int r, int d )
{ return &v.dst_d[d][ ( ( (size_t)s * e.nx + x ) * e.ny + y ) * e.nr + r ]; }

__device__ __forceinline__
uint8_t load_bcs( const Views& v, const Extents& e, int s, int x, int y, int r )
{ return __ldg( &v.bcs[ ( ( (size_t)s * e.nx + x ) * e.ny + y ) * e.nr + r ] ); }

// ----------------------------------------------------------------------------
// Kernel: fast Dirichlet/Neumann matvec
// ----------------------------------------------------------------------------
__global__ __launch_bounds__( TEAM_SIZE, LB_BLOCKS_PER_SM )
inline void epsdivdiv_fast_dn_matvec_kernel( Extents e, Views v )
{
    constexpr double ONE_THIRD      = 1.0 / 3.0;
    constexpr double ONE_SIXTH      = 1.0 / 6.0;
    constexpr double NEG_TWO_THIRDS = -2.0 / 3.0;

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

    int tmp = blockIdx.x;
    const int r_tile_id = tmp % e.r_tiles;   tmp /= e.r_tiles;
    const int lat_y_id  = tmp % e.lat_tiles; tmp /= e.lat_tiles;
    const int lat_x_id  = tmp % e.lat_tiles; tmp /= e.lat_tiles;
    const int s         = tmp;

    const int x0 = lat_x_id * LAT_TILE;
    const int y0 = lat_y_id * LAT_TILE;
    const int r0 = r_tile_id * R_TILE_BLOCK;

    const int tid = threadIdx.x;
    const int tr  = tid % R_TILE;
    const int tx  = ( tid / R_TILE ) % LAT_TILE;
    const int ty  = tid / ( R_TILE * LAT_TILE );

    const int x_cell = x0 + tx;
    const int y_cell = y0 + ty;

    // Shmem partition: coords | src | k | r
    extern __shared__ double smem[];
    double* coords_sh = smem;
    double* src_sh    = coords_sh + NXY * 3;
    double* k_sh      = src_sh + NXY * 3 * NLEV;
    double* r_sh      = k_sh + NXY * NLEV;

    auto sh_coord = [&]( int n, int c ) -> double& { return coords_sh[ n * 3 + c ]; };
    auto sh_src   = [&]( int n, int d, int lvl ) -> double& { return src_sh[ ( n * 3 + d ) * NLEV + lvl ]; };
    auto sh_k     = [&]( int n, int lvl ) -> double& { return k_sh[ n * NLEV + lvl ]; };
    auto node_id  = [&]( int nx, int ny ) -> int { return nx + N_LAT * ny; };

    // ---- cooperative shmem fill ----
    for ( int n = tid; n < NXY; n += TEAM_SIZE )
    {
        const int dxn = n % N_LAT;
        const int dyn = n / N_LAT;
        const int xi  = x0 + dxn;
        const int yi  = y0 + dyn;
        if ( xi <= e.hex_lat && yi <= e.hex_lat )
        {
            sh_coord( n, 0 ) = load_coord( v, e, s, xi, yi, 0 );
            sh_coord( n, 1 ) = load_coord( v, e, s, xi, yi, 1 );
            sh_coord( n, 2 ) = load_coord( v, e, s, xi, yi, 2 );
        }
        else
        {
            sh_coord( n, 0 ) = sh_coord( n, 1 ) = sh_coord( n, 2 ) = 0.0;
        }
    }
    for ( int lvl = tid; lvl < NLEV; lvl += TEAM_SIZE )
    {
        const int rr = r0 + lvl;
        r_sh[ lvl ]  = ( rr <= e.hex_rad ) ? load_radius( v, e, s, rr ) : 0.0;
    }
    const int total_pairs = NXY * NLEV;
    for ( int t = tid; t < total_pairs; t += TEAM_SIZE )
    {
        const int node = t / NLEV;
        const int lvl  = t - node * NLEV;
        const int dxn  = node % N_LAT;
        const int dyn  = node / N_LAT;
        const int xi   = x0 + dxn;
        const int yi   = y0 + dyn;
        const int rr   = r0 + lvl;
        if ( xi <= e.hex_lat && yi <= e.hex_lat && rr <= e.hex_rad )
        {
            sh_k( node, lvl )      = load_k( v, e, s, xi, yi, rr );
            sh_src( node, 0, lvl ) = load_src( v, e, s, xi, yi, rr, 0 );
            sh_src( node, 1, lvl ) = load_src( v, e, s, xi, yi, rr, 1 );
            sh_src( node, 2, lvl ) = load_src( v, e, s, xi, yi, rr, 2 );
        }
        else
        {
            sh_k( node, lvl )      = 0.0;
            sh_src( node, 0, lvl ) = sh_src( node, 1, lvl ) = sh_src( node, 2, lvl ) = 0.0;
        }
    }

    __syncthreads();

    if ( x_cell >= e.hex_lat || y_cell >= e.hex_lat )
        return;

    const int n00 = node_id( tx,     ty     );
    const int n01 = node_id( tx,     ty + 1 );
    const int n10 = node_id( tx + 1, ty     );
    const int n11 = node_id( tx + 1, ty + 1 );

    for ( int pass = 0; pass < R_PASSES; ++pass )
    {
        const int lvl0   = pass * R_TILE + tr;
        const int r_cell = r0 + lvl0;
        if ( r_cell >= e.hex_rad ) break;

        const double r_0 = r_sh[ lvl0     ];
        const double r_1 = r_sh[ lvl0 + 1 ];

        const bool at_cmb     = flag_cmb(     load_bcs( v, e, s, x_cell, y_cell, r_cell ) );
        const bool at_surface = flag_surface( load_bcs( v, e, s, x_cell, y_cell, r_cell + 1 ) );
        const bool at_boundary       = at_cmb || at_surface;
        // BCs are Dirichlet (caller responsibility — matches typical Stokes setup)
        const bool treat_dirichlet   = at_boundary;
        const int  cmb_shift         = ( at_boundary && treat_dirichlet && at_cmb     ) ? 3 : 0;
        const int  surface_shift     = ( at_boundary && treat_dirichlet && at_surface ) ? 3 : 0;

        for ( int w = 0; w < 2; ++w )
        {
            const int v0 = ( w == 0 ) ? n00 : n11;
            const int v1 = ( w == 0 ) ? n10 : n01;
            const int v2 = ( w == 0 ) ? n01 : n10;

            double k_sum = 0.0;
#pragma unroll
            for ( int node = 0; node < 6; ++node )
            {
                const int nid = node_id( tx + WEDGE_NODE_OFF[w][node][0],
                                         ty + WEDGE_NODE_OFF[w][node][1] );
                k_sum += sh_k( nid, lvl0 + WEDGE_NODE_OFF[w][node][2] );
            }
            const double k_eval = ONE_SIXTH * k_sum;

            double kwJ;
            double gu00 = 0.0, gu10 = 0.0, gu11 = 0.0;
            double gu20 = 0.0, gu21 = 0.0, gu22 = 0.0;
            double div_u = 0.0;

            // ===== Phase 1: Jacobian + gather =====
            {
                const double half_dr = 0.5 * ( r_1 - r_0 );
                const double r_mid   = 0.5 * ( r_0 + r_1 );
                const double cv0_0 = sh_coord( v0, 0 ), cv1_0 = sh_coord( v1, 0 ), cv2_0 = sh_coord( v2, 0 );
                const double cv0_1 = sh_coord( v0, 1 ), cv1_1 = sh_coord( v1, 1 ), cv2_1 = sh_coord( v2, 1 );
                const double cv0_2 = sh_coord( v0, 2 ), cv1_2 = sh_coord( v1, 2 ), cv2_2 = sh_coord( v2, 2 );

                const double J00 = r_mid * ( cv1_0 - cv0_0 );
                const double J01 = r_mid * ( cv2_0 - cv0_0 );
                const double J02 = half_dr * ONE_THIRD * ( cv0_0 + cv1_0 + cv2_0 );
                const double J10 = r_mid * ( cv1_1 - cv0_1 );
                const double J11 = r_mid * ( cv2_1 - cv0_1 );
                const double J12 = half_dr * ONE_THIRD * ( cv0_1 + cv1_1 + cv2_1 );
                const double J20 = r_mid * ( cv1_2 - cv0_2 );
                const double J21 = r_mid * ( cv2_2 - cv0_2 );
                const double J22 = half_dr * ONE_THIRD * ( cv0_2 + cv1_2 + cv2_2 );

                const double J_det = J00 * J11 * J22 - J00 * J12 * J21
                                   - J01 * J10 * J22 + J01 * J12 * J20
                                   + J02 * J10 * J21 - J02 * J11 * J20;

                kwJ = k_eval * fabs( J_det );
                const double inv_det = 1.0 / J_det;

                const double i00 = inv_det * (  J11 * J22 - J12 * J21 );
                const double i01 = inv_det * ( -J10 * J22 + J12 * J20 );
                const double i02 = inv_det * (  J10 * J21 - J11 * J20 );
                const double i10 = inv_det * ( -J01 * J22 + J02 * J21 );
                const double i11 = inv_det * (  J00 * J22 - J02 * J20 );
                const double i12 = inv_det * ( -J00 * J21 + J01 * J20 );
                const double i20 = inv_det * (  J01 * J12 - J02 * J11 );
                const double i21 = inv_det * ( -J00 * J12 + J02 * J10 );
                const double i22 = inv_det * (  J00 * J11 - J01 * J10 );

#pragma unroll
                for ( int n = cmb_shift; n < 6 - surface_shift; ++n )
                {
                    const double gx = dN_ref[n][0];
                    const double gy = dN_ref[n][1];
                    const double gz = dN_ref[n][2];
                    const double g0 = i00 * gx + i01 * gy + i02 * gz;
                    const double g1 = i10 * gx + i11 * gy + i12 * gz;
                    const double g2 = i20 * gx + i21 * gy + i22 * gz;

                    const int ddx = WEDGE_NODE_OFF[w][n][0];
                    const int ddy = WEDGE_NODE_OFF[w][n][1];
                    const int ddr = WEDGE_NODE_OFF[w][n][2];
                    const int nid = node_id( tx + ddx, ty + ddy );
                    const int lvl = lvl0 + ddr;

                    const double s0 = sh_src( nid, 0, lvl );
                    const double s1 = sh_src( nid, 1, lvl );
                    const double s2 = sh_src( nid, 2, lvl );

                    gu00  += g0 * s0;
                    gu11  += g1 * s1;
                    gu22  += g2 * s2;
                    gu10  += 0.5 * ( g1 * s0 + g0 * s1 );
                    gu20  += 0.5 * ( g2 * s0 + g0 * s2 );
                    gu21  += 0.5 * ( g2 * s1 + g1 * s2 );
                    div_u += g0 * s0 + g1 * s1 + g2 * s2;
                }
            }

            // ===== Phase 2: recompute Jacobian + scatter =====
            {
                const double half_dr = 0.5 * ( r_1 - r_0 );
                const double r_mid   = 0.5 * ( r_0 + r_1 );
                const double cv0_0 = sh_coord( v0, 0 ), cv1_0 = sh_coord( v1, 0 ), cv2_0 = sh_coord( v2, 0 );
                const double cv0_1 = sh_coord( v0, 1 ), cv1_1 = sh_coord( v1, 1 ), cv2_1 = sh_coord( v2, 1 );
                const double cv0_2 = sh_coord( v0, 2 ), cv1_2 = sh_coord( v1, 2 ), cv2_2 = sh_coord( v2, 2 );

                const double J00 = r_mid * ( cv1_0 - cv0_0 );
                const double J01 = r_mid * ( cv2_0 - cv0_0 );
                const double J02 = half_dr * ONE_THIRD * ( cv0_0 + cv1_0 + cv2_0 );
                const double J10 = r_mid * ( cv1_1 - cv0_1 );
                const double J11 = r_mid * ( cv2_1 - cv0_1 );
                const double J12 = half_dr * ONE_THIRD * ( cv0_1 + cv1_1 + cv2_1 );
                const double J20 = r_mid * ( cv1_2 - cv0_2 );
                const double J21 = r_mid * ( cv2_2 - cv0_2 );
                const double J22 = half_dr * ONE_THIRD * ( cv0_2 + cv1_2 + cv2_2 );

                const double J_det = J00 * J11 * J22 - J00 * J12 * J21
                                   - J01 * J10 * J22 + J01 * J12 * J20
                                   + J02 * J10 * J21 - J02 * J11 * J20;
                const double inv_det = 1.0 / J_det;

                const double i00 = inv_det * (  J11 * J22 - J12 * J21 );
                const double i01 = inv_det * ( -J10 * J22 + J12 * J20 );
                const double i02 = inv_det * (  J10 * J21 - J11 * J20 );
                const double i10 = inv_det * ( -J01 * J22 + J02 * J21 );
                const double i11 = inv_det * (  J00 * J22 - J02 * J20 );
                const double i12 = inv_det * ( -J00 * J21 + J01 * J20 );
                const double i20 = inv_det * (  J01 * J12 - J02 * J11 );
                const double i21 = inv_det * ( -J00 * J12 + J02 * J10 );
                const double i22 = inv_det * (  J00 * J11 - J01 * J10 );

#pragma unroll
                for ( int n = cmb_shift; n < 6 - surface_shift; ++n )
                {
                    const double gx = dN_ref[n][0];
                    const double gy = dN_ref[n][1];
                    const double gz = dN_ref[n][2];
                    const double g0 = i00 * gx + i01 * gy + i02 * gz;
                    const double g1 = i10 * gx + i11 * gy + i12 * gz;
                    const double g2 = i20 * gx + i21 * gy + i22 * gz;

                    const int ddx = WEDGE_NODE_OFF[w][n][0];
                    const int ddy = WEDGE_NODE_OFF[w][n][1];
                    const int ddr = WEDGE_NODE_OFF[w][n][2];

                    atomicAdd( dst_ptr( v, e, s, x_cell + ddx, y_cell + ddy, r_cell + ddr, 0 ),
                               kwJ * ( 2.0 * ( g0 * gu00 + g1 * gu10 + g2 * gu20 )
                                       + NEG_TWO_THIRDS * g0 * div_u ) );
                    atomicAdd( dst_ptr( v, e, s, x_cell + ddx, y_cell + ddy, r_cell + ddr, 1 ),
                               kwJ * ( 2.0 * ( g0 * gu10 + g1 * gu11 + g2 * gu21 )
                                       + NEG_TWO_THIRDS * g1 * div_u ) );
                    atomicAdd( dst_ptr( v, e, s, x_cell + ddx, y_cell + ddy, r_cell + ddr, 2 ),
                               kwJ * ( 2.0 * ( g0 * gu20 + g1 * gu21 + g2 * gu22 )
                                       + NEG_TWO_THIRDS * g2 * div_u ) );
                }

                // Diagonal scatter for boundary nodes the main loop skipped.
                // In matvec mode (Diagonal=false) this fires only on Dirichlet boundary
                // cells: it preserves boundary DoFs by adding their diagonal action.
                if ( treat_dirichlet && at_boundary )
                {
#pragma unroll
                    for ( int n = surface_shift; n < 6 - cmb_shift; ++n )
                    {
                        const double gx = dN_ref[n][0];
                        const double gy = dN_ref[n][1];
                        const double gz = dN_ref[n][2];
                        const double g0 = i00 * gx + i01 * gy + i02 * gz;
                        const double g1 = i10 * gx + i11 * gy + i12 * gz;
                        const double g2 = i20 * gx + i21 * gy + i22 * gz;
                        const double gg = g0 * g0 + g1 * g1 + g2 * g2;

                        const int nid = node_id( tx + WEDGE_NODE_OFF[w][n][0],
                                                 ty + WEDGE_NODE_OFF[w][n][1] );
                        const int lvl = lvl0 + WEDGE_NODE_OFF[w][n][2];
                        const double sv0 = sh_src( nid, 0, lvl );
                        const double sv1 = sh_src( nid, 1, lvl );
                        const double sv2 = sh_src( nid, 2, lvl );

                        const int ddx = WEDGE_NODE_OFF[w][n][0];
                        const int ddy = WEDGE_NODE_OFF[w][n][1];
                        const int ddr = WEDGE_NODE_OFF[w][n][2];

                        atomicAdd( dst_ptr( v, e, s, x_cell + ddx, y_cell + ddy, r_cell + ddr, 0 ),
                                   kwJ * sv0 * ( gg + ONE_THIRD * g0 * g0 ) );
                        atomicAdd( dst_ptr( v, e, s, x_cell + ddx, y_cell + ddy, r_cell + ddr, 1 ),
                                   kwJ * sv1 * ( gg + ONE_THIRD * g1 * g1 ) );
                        atomicAdd( dst_ptr( v, e, s, x_cell + ddx, y_cell + ddy, r_cell + ddr, 2 ),
                                   kwJ * sv2 * ( gg + ONE_THIRD * g2 * g2 ) );
                    }
                }
            }
        }
    }
}

} // namespace cuda_kerngen_detail


// ============================================================================
// Wrapper class with the same constructor signature as EpsilonDivDivKerngen.
// ============================================================================
template < typename ScalarT, int VecDim = 3 >
class EpsilonDivDivKerngenCuda
{
    static_assert( std::is_same_v< ScalarT, double >,
                   "EpsilonDivDivKerngenCuda is double-only in this draft" );
    static_assert( VecDim == 3, "EpsilonDivDivKerngenCuda requires VecDim == 3" );

  public:
    using SrcVectorType = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using DstVectorType = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using ScalarType    = ScalarT;

  private:
    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataScalar< ScalarT >                        k_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;

    int local_subdomains_;
    int hex_lat_;
    int hex_rad_;
    int lat_tiles_;
    int r_tiles_;
    int blocks_;

    linalg::OperatorApplyMode operator_apply_mode_;

  public:
    EpsilonDivDivKerngenCuda(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< ScalarT >&                        radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const grid::Grid4DDataScalar< ScalarT >&                        k,
        BoundaryConditions                                              bcs,
        bool                                                            diagonal,
        linalg::OperatorApplyMode operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode             = linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode operator_stored_matrix_mode = linalg::OperatorStoredMatrixMode::Off )
    : grid_( grid )
    , radii_( radii )
    , k_( k )
    , mask_( mask )
    , operator_apply_mode_( operator_apply_mode )
    {
        if ( diagonal )
            throw std::runtime_error( "EpsilonDivDivKerngenCuda: diagonal=true not supported in this draft" );
        if ( operator_stored_matrix_mode != linalg::OperatorStoredMatrixMode::Off )
            throw std::runtime_error( "EpsilonDivDivKerngenCuda: stored matrices not supported (slow path missing)" );

        const auto cmb_bc = grid::shell::get_boundary_condition_flag( bcs, grid::shell::ShellBoundaryFlag::CMB );
        const auto sfc_bc = grid::shell::get_boundary_condition_flag( bcs, grid::shell::ShellBoundaryFlag::SURFACE );
        if ( cmb_bc == grid::shell::BoundaryConditionFlag::FREESLIP ||
             sfc_bc == grid::shell::BoundaryConditionFlag::FREESLIP )
            throw std::runtime_error( "EpsilonDivDivKerngenCuda: free-slip path missing" );

        const auto& info  = domain.domain_info();
        local_subdomains_ = static_cast< int >( domain.subdomains().size() );
        hex_lat_          = info.subdomain_num_nodes_per_side_laterally() - 1;
        hex_rad_          = info.subdomain_num_nodes_radially() - 1;

        using namespace cuda_kerngen_detail;
        lat_tiles_ = ( hex_lat_ + LAT_TILE - 1 ) / LAT_TILE;
        r_tiles_   = ( hex_rad_ + R_TILE_BLOCK - 1 ) / R_TILE_BLOCK;
        blocks_    = local_subdomains_ * lat_tiles_ * lat_tiles_ * r_tiles_;

        util::logroot << "[EpsDivDivKerngenCuda] tile (x,y,r)=(" << LAT_TILE << "," << LAT_TILE
                      << "," << R_TILE << "), r_passes=" << R_PASSES
                      << ", team_size=" << TEAM_SIZE << ", blocks=" << blocks_ << std::endl;
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, ScalarT( 0 ) );
        }

        using namespace cuda_kerngen_detail;
        Extents e;
        e.hex_lat          = hex_lat_;
        e.hex_rad          = hex_rad_;
        e.local_subdomains = local_subdomains_;
        e.lat_tiles        = lat_tiles_;
        e.r_tiles          = r_tiles_;
        e.nx               = hex_lat_ + 1;
        e.ny               = hex_lat_ + 1;
        e.nr               = hex_rad_ + 1;

        Views v;
        v.coords = grid_.data();
        v.radii  = radii_.data();
        v.k      = k_.data();
        for ( int d = 0; d < 3; ++d )
        {
            v.src_d[d] = src.grid_data().comp_[d].data();
            v.dst_d[d] = dst.grid_data().comp_[d].data();
        }
        v.bcs    = reinterpret_cast< const uint8_t* >( mask_.data() );

        const dim3   grid( static_cast< unsigned >( blocks_ ) );
        const dim3   block( TEAM_SIZE );
        const size_t shmem = shmem_bytes_dn();

        epsdivdiv_fast_dn_matvec_kernel<<< grid, block, shmem, 0 >>>( e, v );
        cudaDeviceSynchronize();
    }
};

} // namespace terra::fe::wedge::operators::shell
