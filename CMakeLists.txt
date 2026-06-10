cmake_minimum_required(VERSION 3.21...4.0)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
  set(SKBUILD_PROJECT_NAME pyrtta)
endif()

if (NOT DEFINED SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION 0.2.1)
endif()

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

if (NOT SKBUILD)
  message(WARNING
    "This project is configured for scikit-build-core. "
    "Use `pip install .` or `python -m build` from the project root.")
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
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

nanobind_add_module(indicator NB_STATIC src/rtta/indicator.cpp)
target_compile_features(indicator PRIVATE cxx_std_23)
target_include_directories(indicator PRIVATE "${FAST_KALMAN_INCLUDE_DIR}")

install(TARGETS indicator LIBRARY DESTINATION rtta)
