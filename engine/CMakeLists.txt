# The 5D chess rules engine (design spec §11.0): a C++20 core library, a C API
# shared library, the test runner and the gate adapter.
#
#   t5d_core   static library - the rules engine itself
#   t5d        shared library - the C API (include/t5d/capi.h), for Python ctypes
#   t5d-tests  the in-house test runner
#   t5d-gate   the differential-testing gate's engine adapter (full and parse mode)
#
# Build with `py engine/build.py [--test]` (see README.md), or by hand:
#   cmake -S engine -B build/engine -G Ninja && cmake --build build/engine
#
# Nothing here may refer to anything under oracles/ (AGENTS.md): the engine's
# build never depends on the test harness or an oracle.

cmake_minimum_required(VERSION 3.20)

project(t5d
  VERSION 0.1.0
  DESCRIPTION "5D chess rules engine"
  LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Build type: Release, Debug, RelWithDebInfo or MinSizeRel" FORCE)
endif()

option(T5D_WERROR "Treat compiler warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Every binary at the top of the build tree.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Release optimises at -O2 (CMake's own default for GCC and Clang is -O3).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  string(REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# ---- options shared by every target ---------------------------------------------

add_library(t5d_options INTERFACE)
target_compile_features(t5d_options INTERFACE cxx_std_20)
if(MSVC)
  target_compile_options(t5d_options INTERFACE /W4 /permissive- /utf-8
    $<$<BOOL:${T5D_WERROR}>:/WX>)
else()
  target_compile_options(t5d_options INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
    $<$<BOOL:${T5D_WERROR}>:-Werror>)
endif()
target_compile_definitions(t5d_options INTERFACE T5D_VERSION_STRING="${PROJECT_VERSION}")

# ---- the core ----------------------------------------------------------------

add_library(t5d_core STATIC
  src/board.cpp
  src/expand.cpp
  src/fen.cpp
  src/fingerprint.cpp
  src/gate.cpp
  src/md5.cpp
  src/move.cpp
  src/movegen.cpp
  src/multiverse.cpp
  src/pgn.cpp
  src/rules.cpp
  src/state.cpp
  src/timeline.cpp
  src/types.cpp
  src/variants.cpp
  src/zobrist.cpp)
target_include_directories(t5d_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(t5d_core PUBLIC t5d_options)
# The core is linked into the shared library, so it must be position-independent.
set_target_properties(t5d_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Stage E4: check status and submission legality, legal actions, mate detection.
target_sources(t5d_core PRIVATE
  src/check.cpp
  src/hypercuboid.cpp
  src/search.cpp)

# ---- the C API shared library -----------------------------------------------------

add_library(t5d SHARED src/capi.cpp)
target_link_libraries(t5d PRIVATE t5d_core)
target_compile_definitions(t5d PRIVATE T5D_BUILDING_DLL)
if(WIN32)
  set_target_properties(t5d PROPERTIES PREFIX "")  # t5d.dll, not libt5d.dll
else()
  # Export the C API only, not the C++ core inside the library.
  set_target_properties(t5d_core t5d PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
endif()

# ---- executables -------------------------------------------------------------

add_executable(t5d-tests
  tests/test_main.cpp
  tests/test_alloc.cpp
  tests/test_apply.cpp
  tests/test_board.cpp
  tests/test_capi.cpp
  tests/test_capi_game.cpp
  tests/test_expand.cpp
  tests/test_expand_reference.cpp
  tests/test_fen.cpp
  tests/test_fingerprint.cpp
  tests/test_gate.cpp
  tests/test_md5.cpp
  tests/test_movegen.cpp
  tests/test_movegen_timelines.cpp
  tests/test_multiverse.cpp
  tests/test_pgn_read.cpp
  tests/test_pgn_replay.cpp
  tests/test_pgn_examples.cpp
  tests/test_random_play.cpp
  tests/test_rules.cpp
  tests/test_special_moves.cpp
  tests/test_state.cpp
  tests/test_synthetic.cpp
  tests/test_timeline.cpp
  tests/test_types.cpp
  tests/test_variants.cpp
  tests/test_zobrist.cpp
  # The C API is tested linked in statically: the same source as the library.
  src/capi.cpp)
target_include_directories(t5d-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(t5d-tests PRIVATE t5d_core t5d_tools)
# Stage E4's tests.
target_sources(t5d-tests PRIVATE
  tests/test_check.cpp
  tests/test_detectors.cpp
  tests/test_search.cpp)

add_executable(t5d-gate tools/t5d_gate.cpp)
target_link_libraries(t5d-gate PRIVATE t5d_core)

# The measurement tools (stage E3) and what they share: SYNTHETIC positions by
# random play (for throughput only), record input, ruleset options.
add_library(t5d_tools STATIC
  tools/synthetic.cpp
  tools/tool_support.cpp)
target_include_directories(t5d_tools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tools)
target_link_libraries(t5d_tools PUBLIC t5d_core)

add_executable(t5d-measure tools/t5d_measure.cpp)
target_link_libraries(t5d-measure PRIVATE t5d_tools)

# The benchmark names its build in its output: the build type and the flags.
find_package(Threads REQUIRED)
add_executable(t5d-bench tools/t5d_bench.cpp)
target_link_libraries(t5d-bench PRIVATE t5d_tools Threads::Threads)
string(TOUPPER "${CMAKE_BUILD_TYPE}" T5D_BUILD_TYPE_UPPER)
string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${T5D_BUILD_TYPE_UPPER}}" T5D_FLAGS_TEXT)
target_compile_definitions(t5d-bench PRIVATE
  "T5D_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\""
  "T5D_CXX_FLAGS=\"${T5D_FLAGS_TEXT}\"")
if(WIN32)
  target_link_libraries(t5d-bench PRIVATE advapi32)  # RegGetValueA: the CPU's name
endif()

# ---- MinGW: no dependence on the toolchain's runtime DLLs --------------------------
#
# With MSYS2's MinGW-w64 g++, the C++ runtime (libstdc++, libgcc) and winpthreads
# are DLLs of the toolchain. Link them in, so that the binaries - and the DLL that
# Python loads - run on a machine (or a PATH) without MSYS2. The DLL needs -static
# as well: with only -static-libgcc -static-libstdc++ it still imports
# libwinpthread-1.dll (seen with `objdump -p t5d.dll`). What remains are system
# DLLs: KERNEL32 and the Universal CRT (api-ms-win-crt-*).

if(MINGW)
  target_link_options(t5d PRIVATE -static-libgcc -static-libstdc++ -static)
  target_link_options(t5d-tests PRIVATE -static)
  target_link_options(t5d-gate PRIVATE -static)
  target_link_options(t5d-measure PRIVATE -static)
  target_link_options(t5d-bench PRIVATE -static)
endif()

# ---- tests -------------------------------------------------------------------

enable_testing()
add_test(NAME t5d-tests COMMAND t5d-tests)
