function(add_terra_performance_benchmark name)
    add_executable(${name} ${name}.cpp)
    if (MPI_FOUND)
        target_link_libraries(${name} terra MPI::MPI_CXX)
    else ()
        target_link_libraries(${name} terra)
    endif ()
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_terra_performance_benchmark(benchmark_alt)
add_terra_performance_benchmark(benchmark_alt_fe)
add_terra_performance_benchmark(benchmark_baseline)
add_terra_performance_benchmark(benchmark_operators)
add_terra_performance_benchmark(benchmark_communication)

# Fused bandwidth benchmark: raw HIP + Kokkos 1D-4D comparison
add_executable(benchmark_bandwidth benchmark_bandwidth.cpp)
target_link_libraries(benchmark_bandwidth Kokkos::kokkos)
target_compile_options(benchmark_bandwidth PRIVATE --offload-arch=gfx90a)
add_test(NAME benchmark_bandwidth COMMAND benchmark_bandwidth)

# Raw HIP bandwidth benchmark (no Kokkos, no terra — hipcc links HIP automatically)
add_executable(benchmark_bandwidth_raw benchmark_bandwidth_raw.cpp)
target_compile_options(benchmark_bandwidth_raw PRIVATE --offload-arch=gfx90a)
add_test(NAME benchmark_bandwidth_raw COMMAND benchmark_bandwidth_raw)

