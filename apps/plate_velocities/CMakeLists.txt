if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/output)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/output")
endif()

add_executable( plate_velocity_demo
                plate_velocity_demo.cpp )

target_include_directories(plate_velocity_demo PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/apps/mantlecirculation/ )

target_link_libraries(plate_velocity_demo
    PRIVATE terra
)

add_executable( plate_velocity_kokkos
                plate_velocity_kokkos.cpp )

target_include_directories(plate_velocity_kokkos PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/apps/mantlecirculation/ )

target_link_libraries(plate_velocity_kokkos
    PRIVATE terra
)

add_executable( plate_bc_demo
                plate_bc_demo.cpp )

target_include_directories(plate_bc_demo PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/apps/mantlecirculation/ )

target_link_libraries(plate_bc_demo
    PRIVATE terra
)

add_executable( plate_velocity_dump plate_velocity_dump.cpp )
target_include_directories(plate_velocity_dump PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/apps/mantlecirculation/ )
target_link_libraries(plate_velocity_dump PRIVATE terra)
