# Global property to collect all test targets.
set_property(GLOBAL PROPERTY TERRA_TEST_TARGETS "")

# Adds a test.
function(add_terra_test name)

    set(num_mpi_procs 1)   # default value

    if (ARGC GREATER 1)
        set(num_mpi_procs ${ARGV1})
    endif ()

    # Check if 'name' is already in the list
    get_property(tests GLOBAL PROPERTY TERRA_TEST_TARGETS)

    list(FIND tests "${name}" index)
    if (index EQUAL -1)
        # Not found

        add_executable(${name} ${name}.cpp)
        if (MPI_FOUND)
            target_link_libraries(${name} terra MPI::MPI_CXX)
        else ()
            target_link_libraries(${name} terra)
        endif ()
        target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)

    endif ()


    add_test(NAME ${name}_np_${num_mpi_procs} COMMAND ${MPIEXEC_EXECUTABLE} -np ${num_mpi_procs} ./${name})

    # Add to global test list
    list(APPEND tests ${name})
    set_property(GLOBAL PROPERTY TERRA_TEST_TARGETS "${tests}")
endfunction()

# Adding all the tests.
add_terra_test(test_arg_parser)
add_terra_test(test_basis_trafo_normal_tangential)
add_terra_test(test_boundary_mass)
add_terra_test(test_debug_assembly)
add_terra_test(test_flag_field_unique_vertices)
add_terra_test(test_heat_eq)
add_terra_test(test_laplace_bicgstab)
add_terra_test(test_laplace_cg)
add_terra_test(test_laplace_cg 2)
add_terra_test(test_laplace_cg 4)
add_terra_test(test_laplace_cg_subdomains)
add_terra_test(test_laplace_compare)
add_terra_test(test_laplace_fgmres)
add_terra_test(test_laplace_jacobi)
add_terra_test(test_laplace_minres)
add_terra_test(test_laplace_multigrid)
add_terra_test(test_laplace_multigrid 2)
add_terra_test(test_laplace_pbicgstab_multigrid)
add_terra_test(test_laplace_richardson)
add_terra_test(test_div_k_grad_cg)
add_terra_test(test_div_k_grad_multigrid_gca)
add_terra_test(test_mpi_2_proc 2)
add_terra_test(test_mpi_smoke)
add_terra_test(test_prolongation_constant)
add_terra_test(test_prolongation_linear)
add_terra_test(test_radial_profiles)
add_terra_test(test_restriction)
add_terra_test(test_stokes_minres)
add_terra_test(test_epsilon_divdiv_stokes)
add_terra_test(test_stokes_pbicgstab_multigrid)
add_terra_test(test_stokes_pbicgstab_multigrid 2)
add_terra_test(test_table)
add_terra_test(test_unsteady_advection_diffusion_supg)
add_terra_test(test_vector_laplace_cg)
add_terra_test(test_epsilon_cg)
add_terra_test(test_epsilon_divdiv_cg)
add_terra_test(test_vector_laplace_compare)
add_terra_test(test_xdmf_writer)

# Adding target for all the tests.
get_property(ALL_TERRA_TESTS GLOBAL PROPERTY TERRA_TEST_TARGETS)
add_custom_target(tests DEPENDS ${ALL_TERRA_TESTS})