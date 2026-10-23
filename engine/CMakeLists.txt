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
# Stage E5a: the legal moves, in tiers.
target_sources(t5d_core PRIVATE src/legal.cpp)
# Stage E6a: the main search (MCTS / MCGS), its built-in evaluators, self-play.
target_sources(t5d_core PRIVATE
  src/mcts.cpp
  src/mcts_evaluators.cpp
  src/selfplay.cpp)

# ---- the C API shared library -----------------------------------------------------

add_library(t5d SHARED src/capi.cpp)
target_link_libraries(t5d PRIVATE t5d_core)
target_sources(t5d PRIVATE src/capi_search.cpp)  # stage E6a: the search's C API
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
  tests/test_gate_exe.cpp
  tests/test_search.cpp)
# D2's flat form: the turn generator (spec 15.2's row).
target_sources(t5d-tests PRIVATE tests/test_turngen.cpp)

# Stage E6a's tests: the search's invariants (§15.4), self-play end to end,
# t5d-play's records through t5d-gate, the search's C API (linked in, as
# capi.cpp is).
target_sources(t5d-tests PRIVATE
  tests/test_capi_search.cpp
  tests/test_mcts.cpp
  tests/test_mcts_memory.cpp
  tests/test_mcts_tree.cpp
  tests/test_play_exe.cpp
  tests/test_selfplay.cpp
  src/capi_search.cpp)
# Register A6 (engine ABI round 1): the turn-state counts of the expansion and
# the C API against a recount from the rulebook, under every ruleset.
target_sources(t5d-tests PRIVATE tests/test_turn_state.cpp)

add_executable(t5d-gate tools/t5d_gate.cpp)
target_link_libraries(t5d-gate PRIVATE t5d_core)

# test_gate_exe.cpp runs the adapter itself: its path, and where the runs'
# files go.
add_dependencies(t5d-tests t5d-gate)
target_compile_definitions(t5d-tests PRIVATE
  "T5D_GATE_EXE=\"$<TARGET_FILE:t5d-gate>\""
  "T5D_TEST_SCRATCH=\"${CMAKE_CURRENT_BINARY_DIR}/test-scratch\"")

# The measurement tools (stage E3) and what they share: SYNTHETIC positions by
# random play (for throughput only), record input, ruleset options.
add_library(t5d_tools STATIC
  tools/synthetic.cpp
  tools/tool_support.cpp)
target_include_directories(t5d_tools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tools)
target_link_libraries(t5d_tools PUBLIC t5d_core)

add_executable(t5d-measure tools/t5d_measure.cpp)
target_link_libraries(t5d-measure PRIVATE t5d_tools)

# Register M22: the turn width of an action (tools/turn_width.*), measured on
# records by t5d-measure --turns / --turn-summary; test_turnwidth.cpp counts
# constructed actions and runs t5d-measure itself on engine/testdata/turn-width.
# Not one of the truth's sources (T5D_TRUTH_SOURCES below).
target_sources(t5d_tools PRIVATE tools/turn_width.cpp)
target_sources(t5d-tests PRIVATE tests/test_turnwidth.cpp)
add_dependencies(t5d-tests t5d-measure)
target_compile_definitions(t5d-tests PRIVATE "T5D_MEASURE_EXE=\"$<TARGET_FILE:t5d-measure>\"")

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

# Stage E5b: the tactical scanner and its tests; experiment 6b's tool.
target_sources(t5d_core PRIVATE src/scanner.cpp)
# Stage E5c: the scanner's Lazy SMP runs threads (std::thread).
target_link_libraries(t5d_core PUBLIC Threads::Threads)
target_sources(t5d-tests PRIVATE tests/test_scanner.cpp)
add_executable(t5d-scan6b tools/t5d_scan6b.cpp)
target_link_libraries(t5d-scan6b PRIVATE t5d_tools Threads::Threads)
target_compile_definitions(t5d-scan6b PRIVATE
  "T5D_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\""
  "T5D_CXX_FLAGS=\"${T5D_FLAGS_TEXT}\"")
if(WIN32)
  target_link_libraries(t5d-scan6b PRIVATE advapi32)  # RegGetValueA: the CPU's name
endif()
if(MINGW)
  target_link_options(t5d-scan6b PRIVATE -static)
endif()

# Stage E8: the verification layer's redesign (M14) - incremental royal-attack
# maps and the proof channel (df-pn over semimoves); 6b'. (Its estimate channel was
# retired at CHECKPOINT 4, verification v3.)
target_sources(t5d_core PRIVATE src/attacks.cpp src/prover.cpp)
target_sources(t5d-tests PRIVATE tests/test_attacks.cpp tests/test_prover.cpp)
# The stores the scanner's and the prover's threads share, taken by many
# threads at once from their first use (register H17: the shards' lock).
target_sources(t5d-tests PRIVATE tests/test_shared_stores.cpp)
add_executable(t5d-scan6bp tools/t5d_scan6bp.cpp)
target_link_libraries(t5d-scan6bp PRIVATE t5d_tools Threads::Threads)
target_compile_definitions(t5d-scan6bp PRIVATE
  "T5D_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\""
  "T5D_CXX_FLAGS=\"${T5D_FLAGS_TEXT}\"")
if(WIN32)
  target_link_libraries(t5d-scan6bp PRIVATE advapi32)  # RegGetValueA: the CPU's name
endif()
if(MINGW)
  target_link_options(t5d-scan6bp PRIVATE -static)
endif()
# test_sixbpp.cpp runs 6b's and 6b''s tools on a record written there: their
# roots must be the ones 6b read (tool_support.hpp, sixb_pgn_options; H51).
add_dependencies(t5d-tests t5d-scan6b t5d-scan6bp)
target_compile_definitions(t5d-tests PRIVATE
  "T5D_SCAN6B_EXE=\"$<TARGET_FILE:t5d-scan6b>\""
  "T5D_SCAN6BP_EXE=\"$<TARGET_FILE:t5d-scan6bp>\"")

# Stage E9: verification v3 (spec 11.4) - Guard, Hunt, the asset map, proof
# certificates checked against E4 alone; Hunt's ground truth and experiment 6b''.
target_sources(t5d_core PRIVATE src/certificate.cpp src/guard.cpp src/hunt.cpp src/assets.cpp)
target_sources(t5d-tests PRIVATE tests/test_certificate.cpp tests/test_guard.cpp tests/test_hunt.cpp
  tests/test_assets.cpp)
# Hunt's ground truth and 6b'': 6b's roots with their next actions, the truth
# file and its SHA-256 pin (shared by the tools and the truth file's test).
target_sources(t5d_tools PRIVATE tools/sha256.cpp tools/sixb_roots.cpp tools/hunt_truth.cpp tools/sixbpp.cpp)
target_sources(t5d-tests PRIVATE tests/test_hunt_truth.cpp tests/test_sixbpp.cpp)
target_compile_definitions(t5d-tests PRIVATE "T5D_ENGINE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
# The truth's code signature (cmake/code_signature.cmake): the sources that
# decide its verdicts - the rules engine, E4, the prover, the certificates, the
# record reader and the recorder.
set(T5D_TRUTH_SOURCES
  include/t5d/attacks.hpp include/t5d/board.hpp include/t5d/certificate.hpp include/t5d/check.hpp
  include/t5d/error.hpp include/t5d/fen.hpp include/t5d/fingerprint.hpp include/t5d/geometry.hpp
  include/t5d/legal.hpp include/t5d/md5.hpp include/t5d/move.hpp include/t5d/movegen.hpp
  include/t5d/multiverse.hpp include/t5d/pgn.hpp include/t5d/prover.hpp include/t5d/rules.hpp
  include/t5d/scanner.hpp include/t5d/search.hpp include/t5d/state.hpp include/t5d/timeline.hpp
  include/t5d/types.hpp include/t5d/variants.hpp include/t5d/zobrist.hpp
  src/attacks.cpp src/board.cpp src/certificate.cpp src/check.cpp src/fen.cpp src/fingerprint.cpp
  src/hypercuboid.cpp src/legal.cpp src/md5.cpp src/move.cpp src/movegen.cpp src/multiverse.cpp
  src/pgn.cpp src/prover.cpp src/rules.cpp src/scanner.cpp src/search.cpp src/search_internal.hpp
  src/state.cpp src/timeline.cpp src/types.cpp src/variants.cpp src/zobrist.cpp
  tools/hunt_truth.hpp tools/hunt_truth.cpp tools/sha256.hpp tools/sha256.cpp tools/sixb_roots.hpp
  tools/sixb_roots.cpp tools/tool_support.hpp tools/tool_support.cpp tools/t5d_hunt_truth.cpp)
set(T5D_SIGNATURE_DEPENDS "")
foreach(f IN LISTS T5D_TRUTH_SOURCES)
  list(APPEND T5D_SIGNATURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
endforeach()
string(REPLACE ";" "," T5D_TRUTH_SOURCES_TEXT "${T5D_TRUTH_SOURCES}")
set(T5D_SIGNATURE_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/t5d_code_signature.hpp")
add_custom_command(
  OUTPUT "${T5D_SIGNATURE_HEADER}"
  COMMAND "${CMAKE_COMMAND}" "-DROOT=${CMAKE_CURRENT_SOURCE_DIR}" "-DFILES=${T5D_TRUTH_SOURCES_TEXT}"
          "-DOUT=${T5D_SIGNATURE_HEADER}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/code_signature.cmake"
  DEPENDS ${T5D_SIGNATURE_DEPENDS} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/code_signature.cmake"
  COMMENT "The truth's code signature"
  VERBATIM)
add_executable(t5d-hunt-truth tools/t5d_hunt_truth.cpp "${T5D_SIGNATURE_HEADER}")
target_include_directories(t5d-hunt-truth PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(t5d-hunt-truth PRIVATE t5d_tools Threads::Threads)
if(MINGW)
  target_link_options(t5d-hunt-truth PRIVATE -static)
endif()
add_executable(t5d-scan6bpp tools/t5d_scan6bpp.cpp "${T5D_SIGNATURE_HEADER}")
target_include_directories(t5d-scan6bpp PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(t5d-scan6bpp PRIVATE t5d_tools Threads::Threads)
target_compile_definitions(t5d-scan6bpp PRIVATE
  "T5D_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\""
  "T5D_CXX_FLAGS=\"${T5D_FLAGS_TEXT}\"")
if(WIN32)
  target_link_libraries(t5d-scan6bpp PRIVATE advapi32)  # RegGetValueA: the CPU's name
endif()
if(MINGW)
  target_link_options(t5d-scan6bpp PRIVATE -static)
endif()

# Stage E6a: t5d-play, the engine against itself, printing the game's 5dpgn
# record; test_play_exe.cpp runs it and replays the record with t5d-gate.
target_sources(t5d_tools PRIVATE tools/play_record.cpp)
add_executable(t5d-play tools/t5d_play.cpp)
target_link_libraries(t5d-play PRIVATE t5d_tools)
add_dependencies(t5d-tests t5d-play)
target_compile_definitions(t5d-tests PRIVATE "T5D_PLAY_EXE=\"$<TARGET_FILE:t5d-play>\"")
if(MINGW)
  target_link_options(t5d-play PRIVATE -static)
endif()

# D2's tree form against the ply split on recorded games' turn starts:
# evaluation counts (docs/MEASUREMENTS.md). Not one of the truth's sources.
add_executable(t5d-tree-count tools/t5d_tree_count.cpp)
target_link_libraries(t5d-tree-count PRIVATE t5d_tools)
if(MINGW)
  target_link_options(t5d-tree-count PRIVATE -static)
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
