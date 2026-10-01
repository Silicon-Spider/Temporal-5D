# Test-time driver for the pinned ftxi/5dchess_engine oracle.
#
# Configured by the harness (oracle_harness/build.py) with FTXI_SOURCE_DIR set to
# an export of the locked commit. The oracle's own CMake defines its libraries;
# nothing here reimplements or patches them.
cmake_minimum_required(VERSION 3.20)
project(ftxi_oracle_driver LANGUAGES CXX)

set(FTXI_SOURCE_DIR "" CACHE PATH "Exported tree of the pinned ftxi/5dchess_engine commit")
if(NOT EXISTS "${FTXI_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "FTXI_SOURCE_DIR must point at the exported pinned tree (got '${FTXI_SOURCE_DIR}')")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory("${FTXI_SOURCE_DIR}" ftxi EXCLUDE_FROM_ALL)

# search_tools.cpp is the oracle's own action counter (the one behind
# `5dtools count`); compiling it here keeps the counts identical to the stock tool.
# It is copied byte for byte into the build tree first: compiled in place, its
# object file would be named after its absolute path and overflow MAX_PATH on Windows.
configure_file("${FTXI_SOURCE_DIR}/tools/search_tools.cpp"
               "${CMAKE_BINARY_DIR}/oracle_search_tools.cpp" COPYONLY)
add_executable(ftxi_oracle_driver
    ftxi_driver.cpp
    "${CMAKE_BINARY_DIR}/oracle_search_tools.cpp"
)
target_include_directories(ftxi_oracle_driver PRIVATE "${FTXI_SOURCE_DIR}/tools")
target_link_libraries(ftxi_oracle_driver PRIVATE 5dchess_core)

# Same constexpr limits the oracle sets for its own translation units.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(ftxi_oracle_driver PRIVATE -fconstexpr-steps=100000000)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(ftxi_oracle_driver PRIVATE
        -fconstexpr-depth=1000000
        -fconstexpr-ops-limit=100000000
    )
elseif(MSVC)
    target_compile_options(ftxi_oracle_driver PRIVATE /permissive- /constexpr:steps100000000)
endif()

set_target_properties(ftxi_oracle_driver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
