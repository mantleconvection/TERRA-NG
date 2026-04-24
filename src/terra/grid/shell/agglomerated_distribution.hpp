
#pragma once

#include "grid/shell/spherical_shell.hpp"
#include "terra/mpi/level_comms.hpp"

namespace terra::grid::shell {

/// @brief Wrap an existing subdomain_to_rank function to emit sub-comm-local ranks.
///
/// If the original function assigns subdomain s to parent-comm rank r, the wrapped
/// function returns mpi::agglomerate_rank(r, factor), i.e. the corresponding rank on
/// the factor-reduced sub-comm built by mpi::build_level_comms.
///
/// Assumes the sub-comm was produced with the same contiguous-stride rule (every
/// factor-th parent rank kept), which is how build_level_comms constructs them.
///
/// @param orig    Original assignment against the parent comm.
/// @param factor  Agglomeration factor for this descent step.
/// @return        New assignment function targeting the sub-comm.
inline SubdomainToRankDistributionFunction
    agglomerated_subdomain_to_rank( SubdomainToRankDistributionFunction orig, int factor )
{
    return [orig = std::move( orig ), factor](
               const SubdomainInfo& s,
               const int            num_subdomains_per_diamond_side,
               const int            num_subdomains_in_radial_direction ) -> mpi::MPIRank {
        const auto parent_rank = orig( s, num_subdomains_per_diamond_side, num_subdomains_in_radial_direction );
        return mpi::agglomerate_rank( parent_rank, factor );
    };
}

/// @brief Compose a chain of agglomerated subdomain_to_rank functions for a level ladder.
///
/// Given an original function for the finest level and a list of descent factors
/// (the same vector passed to build_level_comms), returns a per-level subdomain_to_rank
/// list with index i corresponding to level_comms[i]. Level 0 is the finest (original
/// function unchanged); each later level composes one more factor division.
///
/// @param orig    Subdomain assignment on the finest comm.
/// @param factors Per-descent agglomeration factors.
/// @return        Per-level assignment functions. Length = factors.size() + 1.
inline std::vector< SubdomainToRankDistributionFunction > build_level_subdomain_to_rank(
    const SubdomainToRankDistributionFunction& orig,
    const std::vector< int >&                  factors )
{
    std::vector< SubdomainToRankDistributionFunction > result;
    result.reserve( factors.size() + 1 );
    result.push_back( orig );

    int cumulative_factor = 1;
    for ( const int f : factors )
    {
        cumulative_factor *= f;
        result.push_back( agglomerated_subdomain_to_rank( orig, cumulative_factor ) );
    }
    return result;
}

} // namespace terra::grid::shell
