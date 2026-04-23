add_library(terra)

target_link_libraries(terra PUBLIC Kokkos::kokkos MPI::MPI_CXX)
target_include_directories(terra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(terra
        PRIVATE
        library_dummy.cpp
        types.hpp)

if(NESMIK_HOME)
	target_include_directories(terra PUBLIC ${NESMIK_HOME}/include)
	target_link_directories(terra PUBLIC ${NESMIK_HOME}/lib ${NESMIK_HOME}/lib64)
	target_link_libraries(terra PUBLIC nesmik)
        target_compile_definitions(terra PUBLIC TERRANEO_USE_NESMIK)
endif()

add_subdirectory(communication)
add_subdirectory(dense)
add_subdirectory(eigen)
add_subdirectory(fe)
add_subdirectory(fv)
add_subdirectory(geophysics)
add_subdirectory(grid)
add_subdirectory(io)
add_subdirectory(kokkos)
add_subdirectory(linalg)
add_subdirectory(mpi)
add_subdirectory(shell)
add_subdirectory(util)