
#pragma once

#include <cassert>
#include <stdexcept>
#include <vector>

#include "terra/mpi/mpi.hpp"

namespace terra::mpi {

/// @brief Build a nested ladder of sub-communicators for hierarchical agglomeration.
///
/// Given a parent communicator (typically MPI_COMM_WORLD) and per-level
/// agglomeration factors, constructs a sequence of communicators where each
/// successor is a subset of the preceding one. Used by multigrid to run coarse
/// levels on fewer ranks so collectives cost less at the bottom of the V-cycle.
///
/// Factor semantics: at descent step i, ranks whose index on the parent comm
/// is divisible by factors[i] are kept, the rest drop out and get
/// MPI_COMM_NULL at that and every deeper level.
///
/// The result has size factors.size() + 1: index 0 is the finest (= root),
/// the rest are progressively smaller. The sizes on surviving ranks are
/// {N, N/factors[0], N/(factors[0]*factors[1]), ...}.
///
/// @param root_comm   The finest-level communicator.
/// @param factors     Per-descent agglomeration factors. Must all be >= 1 and
///                    evenly divide the preceding level size.
/// @return            Per-level communicators. Entries past the first drop-out
///                    are MPI_COMM_NULL for dropped ranks.
inline std::vector< MPI_Comm > build_level_comms( MPI_Comm root_comm, const std::vector< int >& factors )
{
    std::vector< MPI_Comm > level_comms;
    level_comms.reserve( factors.size() + 1 );
    level_comms.push_back( root_comm );

    MPI_Comm cur = root_comm;

    for ( size_t i = 0; i < factors.size(); ++i )
    {
        const int factor = factors[i];
        if ( factor < 1 )
        {
            throw std::runtime_error( "build_level_comms: agglomeration factor must be >= 1" );
        }

        if ( cur == MPI_COMM_NULL )
        {
            // This rank has already dropped out above.
            level_comms.push_back( MPI_COMM_NULL );
            continue;
        }

        const int parent_size = num_processes( cur );
        const int parent_rank = rank( cur );

        if ( parent_size % factor != 0 )
        {
            throw std::runtime_error(
                "build_level_comms: parent comm size must be a multiple of the agglomeration factor" );
        }

        // Keep ranks whose parent index is divisible by `factor` (contiguous stride).
        // Key preserves the ordering so that sub-comm rank == parent_rank / factor.
        const int color = ( parent_rank % factor == 0 ) ? 0 : MPI_UNDEFINED;
        const int key   = parent_rank / factor;

        MPI_Comm sub = MPI_COMM_NULL;
        MPI_Comm_split( cur, color, key, &sub );
        level_comms.push_back( sub );

        cur = sub;
    }

    return level_comms;
}

/// @brief Translate a parent-comm rank into its agglomerated sub-comm rank.
///
/// Inverse of the color/key rule in build_level_comms: if agglomeration groups
/// every `factor` consecutive parent ranks into a single sub-comm rank, then
/// parent rank r maps to sub-comm rank r / factor.
///
/// Useful when remapping a subdomain_to_rank function to the coarser comm.
///
/// @param parent_rank Rank on the parent comm.
/// @param factor      Agglomeration factor at this descent step.
/// @return            Rank on the sub-comm (valid only if parent_rank % factor == 0).
inline MPIRank agglomerate_rank( MPIRank parent_rank, int factor )
{
    assert( factor >= 1 );
    return parent_rank / factor;
}

/// @brief Free a level-comm ladder. Call once at MG teardown.
///
/// MPI_Comm_free is collective on the comm, so every rank that got a non-null
/// handle must call it. Ranks that got MPI_COMM_NULL at a level must skip.
/// The root comm (index 0) is not freed here — caller owns it.
inline void free_level_comms( std::vector< MPI_Comm >& level_comms )
{
    for ( size_t i = 1; i < level_comms.size(); ++i )
    {
        if ( level_comms[i] != MPI_COMM_NULL )
        {
            MPI_Comm_free( &level_comms[i] );
            level_comms[i] = MPI_COMM_NULL;
        }
    }
}

} // namespace terra::mpi
