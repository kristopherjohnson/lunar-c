# To use CMake to build and run lunar, execute these commands:
#
# mkdir build
# cd build
# cmake ..
# cmake --build .
# ./lunar

cmake_minimum_required(VERSION 3.0.0)
project(lunar VERSION 1.0.0)

include(CTest)
enable_testing()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # using Clang or AppleClang
    add_compile_options(-Wall -O3)
    link_libraries(-lm)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # using GCC
    add_compile_options(-Wall -O3)
    link_libraries(-lm)
endif()

add_executable(lunar lunar.c)

# CTest
#
# Run 'ctest' in the build directory to run the tests.

set(TEST_DATA_DIR "${PROJECT_SOURCE_DIR}/test")

find_program(DIFF diff)
find_program(SHELL NAMES bash zsh sh)

add_test(NAME test_good
    COMMAND ${SHELL} -c "$<TARGET_FILE:lunar> --echo < ${TEST_DATA_DIR}/good_input.txt > good_output.txt && ${DIFF} ${TEST_DATA_DIR}/good_output_expected.txt good_output.txt")

add_test(NAME test_success
    COMMAND ${SHELL} -c "$<TARGET_FILE:lunar> --echo < ${TEST_DATA_DIR}/success_input.txt > success_output.txt && ${DIFF} ${TEST_DATA_DIR}/success_output_expected.txt success_output.txt")

add_test(NAME test_failure
    COMMAND ${SHELL} -c "$<TARGET_FILE:lunar> --echo < ${TEST_DATA_DIR}/failure_input.txt > failure_output.txt && ${DIFF} ${TEST_DATA_DIR}/failure_output_expected.txt failure_output.txt")

# CPack

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)