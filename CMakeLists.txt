# To use CMake to build and run lunar, execute these commands:
#
# mkdir build
# cd build
# cmake ..
# cmake --build
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

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
