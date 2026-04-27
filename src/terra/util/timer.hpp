#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#ifdef TERRANEO_USE_NESMIK
#include <nesmik/nesmik.hpp>
#endif

namespace terra::util {

/// @brief Node representing a timed region in the hierarchy.
///
/// @note See class `Timer` for actually running a timer.
class TimerNode
{
    std::string                                           name;              ///< Name of the timer region
    double                                                total_time{ 0.0 }; ///< Accumulated time (per rank)
    int                                                   count{ 0 };        ///< Number of times this node was timed
    std::map< std::string, std::shared_ptr< TimerNode > > children;          ///< Nested child timers
    TimerNode*                                            parent{ nullptr }; ///< Parent node pointer

    // Aggregated statistics across MPI ranks
    double root_time{ 0.0 }, sum_time{ 0.0 }, min_time{ 0.0 }, max_time{ 0.0 }, avg_time{ 0.0 };

  public:
    friend class TimerTree;

    /// @brief Constructor
    TimerNode( const std::string& n, TimerNode* p = nullptr )
    : name( n )
    , parent( p )
    {}

    void clear_this_and_children()
    {
        total_time = 0.0;
        count      = 0.0;
        root_time  = 0.0;
        sum_time   = 0.0;
        min_time   = 0.0;
        max_time   = 0.0;
        avg_time   = 0.0;
        children.clear();
    }

    /// @brief Convert this node (and children) to JSON (per-rank)
    std::string to_json( int indent = 0 ) const
    {
        std::ostringstream oss;
        std::string        pad( indent, ' ' );
        oss << pad << "{\n";
        oss << pad << "  \"name\": \"" << name << "\",\n";
        oss << pad << "  \"total_time\": " << total_time << ",\n";
        oss << pad << "  \"count\": " << count << ",\n";
        oss << pad << "  \"children\": [\n";
        int i = 0;
        for ( const auto& child : children | std::ranges::views::values )
        {
            oss << child->to_json( indent + 4 );
            if ( i + 1 < children.size() )
            {
                oss << ",";
            }
            oss << "\n";
            i++;
        }
        oss << pad << "  ]\n" << pad << "}";
        return oss.str();
    }

    /// @brief Convert this node (and children) to JSON with MPI-aggregated statistics
    std::string to_agg_json( int indent = 0 ) const
    {
        std::ostringstream oss;
        std::string        pad( indent, ' ' );
        oss << pad << "{\n";
        oss << pad << "  \"name\": \"" << name << "\",\n";
        oss << pad << "  \"root_time\": " << root_time << ",\n";
        oss << pad << "  \"sum_time\": " << sum_time << ",\n";
        oss << pad << "  \"min_time\": " << min_time << ",\n";
        oss << pad << "  \"avg_time\": " << avg_time << ",\n";
        oss << pad << "  \"max_time\": " << max_time << ",\n";
        oss << pad << "  \"count\": " << count << ",\n";
        oss << pad << "  \"children\": [\n";
        int i = 0;
        for ( const auto& child : children | std::ranges::views::values )
        {
            oss << child->to_agg_json( indent + 4 );
            if ( i + 1 < children.size() )
            {
                oss << ",";
            }
            oss << "\n";
            i++;
        }
        oss << pad << "  ]\n" << pad << "}";
        return oss.str();
    }
};

/// @brief Singleton tree managing all timer nodes per MPI rank
///
/// @note Use `Timer` class for the actually starting and stopping timers. Internally `Timer` objects will access a
///       `TimerTree` singleton. So you can easily add timer calls without changing the API of your code.
///
/// Can be exported via json.
///
/// Example:
/// @code
/// auto tt = TimerTree::instance();
///
/// tt.aggregate_mpi();
/// std::cout << tt.json() << std::endl;
/// std::cout << tt.json_aggregate() << std::endl;
/// tt.clear();
/// @endcode
///
/// Example output for `json()`.
/// Note that the root node will always be there carrying no timings.
/// @code
/// {
///   "name": "root",
///   "total_time": 0,
///   "count": 0,
///   "children": [
///     {
///       "name": "laplace_apply",
///       "total_time": 0.356301,
///       "count": 28,
///       "children": [
///         {
///           "name": "laplace_comm",
///           "total_time": 0.02748,
///           "count": 28,
///           "children": [
///           ]
///         },
///         {
///           "name": "laplace_kernel",
///           "total_time": 0.327421,
///           "count": 28,
///           "children": [
///           ]
///         }
///       ]
///     }
///   ]
/// }
/// @endcode
class TimerTree
{
    TimerNode  root{ "root" };   ///< Root node
    TimerNode* current{ &root }; ///< Pointer to current active node
    std::mutex mtx;              ///< Mutex for thread safety

  public:
    /// @brief Access the singleton instance
    static TimerTree& instance()
    {
        static TimerTree tree;
        return tree;
    }

    void clear()
    {
        std::lock_guard< std::mutex > lock( mtx );
        root.clear_this_and_children();
        current = &root;
    }

    /// @brief Enter a new timing scope
    void enter_scope( const std::string& name )
    {
        std::lock_guard< std::mutex > lock( mtx );
        if ( !current->children.contains( name ) )
        {
            current->children[name] = std::make_shared< TimerNode >( name, current );
        }
        current = current->children[name].get();
    }

    /// @brief Exit the current timing scope and record elapsed time
    void exit_scope( double elapsed )
    {
        std::lock_guard< std::mutex > lock( mtx );
        current->total_time += elapsed;
        current->count += 1;
        if ( current->parent )
        {
            current = current->parent;
        }
    }

    /// @brief Per-rank json tree.
    ///
    /// Returns a definitely non-reduced timer tree in json format.
    /// This means that this returns the process-local timings depending on the process that calls this method.
    std::string json() { return root.to_json(); }

    /// @brief MPI-reduced / aggregate json.
    ///
    /// Returns the timings after reduction over all processes.
    /// You need to call aggregate_mpi() before this for reasonable results.
    ///
    /// This method does not need to be called collectively.
    std::string json_aggregate() { return root.to_agg_json(); }

    /// @brief Aggregate timings across all MPI ranks
    ///
    /// Must be called collectively.
    void aggregate_mpi() { aggregate_node( &root, MPI_COMM_WORLD ); }

  private:
    /// @brief Recursively aggregate a node's timings across MPI ranks
    void aggregate_node( TimerNode* node, MPI_Comm comm )
    {
        double local_time = node->total_time;
        double root_time, min_time, max_time, sum_time;

        root_time = local_time;
        MPI_Bcast( &root_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
        MPI_Allreduce( &local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, comm );
        MPI_Allreduce( &local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, comm );
        MPI_Allreduce( &local_time, &sum_time, 1, MPI_DOUBLE, MPI_SUM, comm );

        int size;
        MPI_Comm_size( comm, &size );
        node->root_time = root_time;
        node->sum_time  = sum_time;
        node->min_time  = min_time;
        node->max_time  = max_time;
        node->avg_time  = sum_time / size;

        for ( auto& child : node->children | std::ranges::views::values )
        {
            aggregate_node( child.get(), comm );
        }
    }
};

/// @brief Timer supporting RAII scope or manual stop.
///
/// Starts timer on construction.
///
/// Automatically adds timing to `TimerTree`'s singleton instance.
/// See `TimerTree` for details on how to export the timings.
///
/// Example usage: scoped
/// @code
/// {
///     Timer t("compute"); // scoped timer - starts here
///     // do computation
/// } // timer ends here - writes result to TimerTree::instance()
/// @endcode
///
/// Example usage: stop explicitly
/// @code
/// {
///     Timer t("compute"); // scoped timer - starts here
///     // do computation
///     t.stop() // timer ends here - writes result to TimerTree::instance()
///     // do something that is not included in timing
/// }
/// @endcode
///
class Timer
{
    std::string   name;             ///< Timer name
    Kokkos::Timer timer;            ///< Underlying Kokkos timer
    bool          running{ false }; ///< Is timer currently running

  public:
    /// @brief Constructor - starts the timer
    /// @param n Timer name
    explicit Timer( const std::string& n )
    : name( n )
    {
        TimerTree::instance().enter_scope( name );
        #ifdef TERRANEO_USE_NESMIK
        nesmik::region_start( name );
        #endif
        timer.reset();
        running = true;
    }

    /// @brief Stop the timer and record elapsed time.
    ///
    /// Can be safely called twice - does not do anything on second call.
    void stop()
    {
        if ( running )
        {
            double elapsed = timer.seconds();
            TimerTree::instance().exit_scope( elapsed );
            #ifdef TERRANEO_USE_NESMIK
            nesmik::region_stop( name );
            #endif
            running = false;
        }
    }

    /// @brief Destructor stops timer if still running.
    ///
    /// Can be used instead of stopping manually.
    ~Timer()
    {
        if ( running )
        {
            stop();
        }
    }
};

} // namespace terra::util
