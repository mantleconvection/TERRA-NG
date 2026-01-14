cmake_minimum_required(VERSION 3.21)

# Project name and version
project(TerraNeo VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable verbose makefile output
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Find Kokkos
# find_package(Kokkos REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH})
    foreach (LIB ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
        if (LIB)
            list(APPEND SERVICE_LIBS ${LIB})
        endif ()
    endforeach ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_COMPILE_FLAGS}")

    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
endif ()

set(EIGEN_DIR extern/eigen-3.4.0 CACHE PATH "eigen path")
include_directories(SYSTEM ${EIGEN_DIR})
link_directories(${EIGEN_DIR})

# Enable testing
enable_testing()

# Add the src and tests subdirectories
add_subdirectory(apps)
add_subdirectory(extern)
add_subdirectory(src)
add_subdirectory(tests)

# A custom target to be built by the CI.
# Mostly to limit compile time and still cover a bit of everything.
add_custom_target(ci-subset
        DEPENDS
        mantlecirculation
        benchmark_baseline
        benchmark_operators
        read_checkpoint
        visualize_radial_profile
        visualize_spherical_harmonics
        visualize_thick_spherical_shell_mesh
        test_laplace_multigrid
        test_epsilon_divdiv_stokes
        test_unsteady_advection_diffusion_supg)