cmake_minimum_required(VERSION 3.21...4.0)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
  set(SKBUILD_PROJECT_NAME rtta)
endif()

if (NOT DEFINED SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION 0.2.0)
endif()

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

if (NOT SKBUILD)
  message(WARNING
    "This project is configured for scikit-build-core. "
    "Use `pip install .` or `python -m build` from the project root.")
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(indicator NB_STATIC src/rtta/indicator.cpp)
target_compile_features(indicator PRIVATE cxx_std_23)

install(TARGETS indicator LIBRARY DESTINATION rtta)
