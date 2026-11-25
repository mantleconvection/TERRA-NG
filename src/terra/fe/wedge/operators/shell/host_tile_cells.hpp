#pragma once

/// @file  host_tile_cells.hpp
/// @brief Runs the team-tiled fast operator paths on a multicore CPU.
///
/// The fast paths of the kerngen operators stage one tile of the mesh (coordinates, radii,
/// source values) in team scratch, and each team thread then computes one cell of that tile.
/// On a host backend (Serial, OpenMP) a team is launched with a single thread instead, and
/// that thread walks every cell of its tile. The staged tile is reused for all of its cells,
/// the per-cell arithmetic is the same as on a device, and the tiles are spread over the
/// cores (OpenMP) or the MPI ranks (Serial).

#include <type_traits>

#include <Kokkos_Core.hpp>

namespace terra::fe::wedge::operators::shell {

/// True if the kernels run on the CPU (the default execution space is a host backend).
inline constexpr bool default_space_is_host =
    std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace >;

/// Team size to launch for a tile of `cells` cells: one thread per cell on a device, one thread per tile on the host.
constexpr int tile_team_size( const int cells )
{
    return default_space_is_host ? 1 : cells;
}

/// Calls `f( cell )` for the tile cells owned by the calling team thread: the cell with its team rank on a device,
/// all `cells` cells of the tile on the host.
template < typename Team, typename F >
KOKKOS_FORCEINLINE_FUNCTION void for_each_tile_cell( const Team& team, const int cells, F&& f )
{
    if constexpr ( default_space_is_host )
    {
        for ( int cell = 0; cell < cells; ++cell )
            f( cell );
    }
    else
    {
        f( team.team_rank() );
    }
}

} // namespace terra::fe::wedge::operators::shell
