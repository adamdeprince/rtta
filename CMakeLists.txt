cmake_minimum_required(VERSION 3.21...4.0)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
  set(SKBUILD_PROJECT_NAME pyrtta)
endif()

if (NOT DEFINED SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION 0.2.2)
endif()

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

if (NOT SKBUILD)
  message(WARNING
    "This project is configured for scikit-build-core. "
    "Use `pip install .` or `python -m build` from the project root.")
endif()

# Development.SABIModule is required for nanobind STABLE_ABI / abi3 wheels (CPython 3.12+).
find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import importlib.metadata as m; print(m.distribution('fast-kalman').locate_file('include'))"
  OUTPUT_VARIABLE FAST_KALMAN_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE FAST_KALMAN_INCLUDE_RESULT
)
if (NOT FAST_KALMAN_INCLUDE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "fast-kalman==0.2.3 must be installed in the build environment. "
    "It is listed in pyproject.toml build-system.requires.")
endif()
if (NOT EXISTS "${FAST_KALMAN_INCLUDE_DIR}/kalman/kalman.hpp")
  message(FATAL_ERROR
    "fast-kalman was found, but ${FAST_KALMAN_INCLUDE_DIR}/kalman/kalman.hpp does not exist.")
endif()

# STABLE_ABI + scikit-build wheel.py-api=cp312 produces one abi3 wheel for CPython 3.12+.
nanobind_add_module(indicator NB_STATIC STABLE_ABI src/rtta/indicator.cpp)
target_compile_features(indicator PRIVATE cxx_std_23)
target_include_directories(indicator PRIVATE "${FAST_KALMAN_INCLUDE_DIR}")

# Release micro-opts for the incremental update hot path.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
target_compile_options(indicator PRIVATE
  $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -DNDEBUG -ffp-contract=off>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG>
  >
)
include(CheckIPOSupported)
check_ipo_supported(RESULT RTTA_IPO_SUPPORTED OUTPUT RTTA_IPO_ERROR)
if (RTTA_IPO_SUPPORTED)
  set_property(TARGET indicator PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  set_property(TARGET indicator PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
endif()

install(TARGETS indicator LIBRARY DESTINATION rtta)
