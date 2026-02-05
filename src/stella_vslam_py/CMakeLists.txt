find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

set(PYBIND_PROJECT_NAME stellapy)
pybind11_add_module(${PYBIND_PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/stellapy.cc)
target_link_libraries(${PYBIND_PROJECT_NAME} PUBLIC ${PROJECT_NAME})
set_target_properties(${PYBIND_PROJECT_NAME} PROPERTIES
                      ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
                      LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

install(TARGETS ${PYBIND_PROJECT_NAME}
        LIBRARY DESTINATION ${Python_SITEARCH}
        ARCHIVE DESTINATION ${Python_SITEARCH})

# Optional stub generation
pybind11_find_import(pybind11_stubgen)
if(Python_Interpreter_FOUND AND pybind11_stubgen_FOUND)
    add_custom_command(
        OUTPUT ${PROJECT_BINARY_DIR}/lib/${PYBIND_PROJECT_NAME}.pyi
        COMMAND ${Python_EXECUTABLE} -m pybind11_stubgen -o ${PROJECT_BINARY_DIR}/lib ${PYBIND_PROJECT_NAME}
        DEPENDS ${PYBIND_PROJECT_NAME}
        COMMENT "Generating Python stubs"
    )
    add_custom_target(${PYBIND_PROJECT_NAME}_stubs ALL DEPENDS ${PROJECT_BINARY_DIR}/lib/${PYBIND_PROJECT_NAME}.pyi)

    install(FILES ${PROJECT_BINARY_DIR}/lib/${PYBIND_PROJECT_NAME}.pyi
            DESTINATION ${Python_SITEARCH})
endif()
