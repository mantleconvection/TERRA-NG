

#pragma once
#include "kokkos/kokkos_wrapper.hpp"
#include "mpi/mpi.hpp"

namespace terra::util {

namespace detail {

class KokkosContext
{
  public:
    KokkosContext( const KokkosContext& )            = delete;
    KokkosContext& operator=( const KokkosContext& ) = delete;
    KokkosContext( KokkosContext&& )                 = delete;
    KokkosContext& operator=( KokkosContext&& )      = delete;

    static void initialize( int& argc, char** argv ) { instance( argc, argv ); }

  private:
    bool kokkos_initialized_ = false;

    // private constructor
    KokkosContext( int& argc, char** argv )
    {
        if ( Kokkos::is_initialized() )
        {
            throw std::runtime_error( "Kokkos already initialized!" );
        }

        Kokkos::initialize( argc, argv );

        kokkos_initialized_ = true;
    }

    // private destructor
    ~KokkosContext()
    {
        if ( kokkos_initialized_ && !Kokkos::is_finalized() )
        {
            Kokkos::finalize();
        }
    }

    // singleton instance accessor
    static KokkosContext& instance( int& argc, char** argv )
    {
        static KokkosContext guard( argc, argv );
        return guard;
    }
};

} // namespace detail

/// @brief RAII approach to safely initialize MPI and Kokkos.
///
/// At the start of your main(), call this function to fire up MPI and Kokkos:
///
///      int main( int argc, char** argv)
///      {
///          terra::util::terra_initialize( &argc, &argv );
///
///          // Here goes your cool app/test. MPI and Kokkos are finalized automatically (even if you
///          // throw an exception).
///
///          // ...
///      } // no need to call any finalisation; destructors of internal objects handle shutdown
inline void terra_initialize( int* argc, char*** argv )
{
    // The order is important!
    mpi::detail::MPIContext::initialize( argc, argv );
    detail::KokkosContext::initialize( *argc, *argv );
}

} // namespace terra::util
