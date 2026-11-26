target_sources(terra
        PRIVATE
        integrands.hpp)

add_subdirectory(operators)
add_subdirectory(quadrature)
add_subdirectory(shell)
add_subdirectory(sl)
