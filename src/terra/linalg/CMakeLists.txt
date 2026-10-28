target_sources(terra
        PRIVATE
        linalg.hpp
        operator.hpp
        diagonally_scaled_operator.hpp
        vector.hpp
        vector_w1.hpp
        vector_w1_iso_w2_w1.hpp)

add_subdirectory(solvers)
add_subdirectory(trafo)
add_subdirectory(util)
