if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/output)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/output")
endif()

add_executable( plate_velocity_demo
                plate_velocity_demo.cpp )

target_include_directories(plate_velocity_demo PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/apps/mantlecirculation/ )

target_link_libraries(plate_velocity_demo
    PRIVATE terra
)