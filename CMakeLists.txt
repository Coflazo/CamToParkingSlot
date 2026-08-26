cmake_minimum_required(VERSION 3.24)
project(parkfit_nl LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(PARKFIT_BUILD_TESTS    "Build the C++ test suite"           ON)
option(PARKFIT_BUILD_VISION   "Build the camera vision worker"     ON)
option(PARKFIT_BUILD_BINDINGS "Build the Python extension module"  ON)

# MSVC needs to be told, in 2026, that it should report the real __cplusplus value and
# not define min/max macros that collide with std::min and std::max.
if(MSVC)
  add_compile_options(/W4 /permissive- /Zc:__cplusplus /utf-8 /EHsc)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# parkfit_core -- header-only geometry, indexing, fit and ranking.
# ---------------------------------------------------------------------------
add_library(parkfit_core INTERFACE)
target_include_directories(parkfit_core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/include)
target_compile_features(parkfit_core INTERFACE cxx_std_20)

# ---------------------------------------------------------------------------
# parkfit_vision -- frame health, calibration, gap estimation, temporal state.
# ---------------------------------------------------------------------------
if(PARKFIT_BUILD_VISION)
  file(GLOB PARKFIT_VISION_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/vision/src/*.cpp)
  if(PARKFIT_VISION_SOURCES)
    add_library(parkfit_vision STATIC ${PARKFIT_VISION_SOURCES})
  else()
    add_library(parkfit_vision INTERFACE)
  endif()
  if(TARGET parkfit_vision AND PARKFIT_VISION_SOURCES)
    target_include_directories(parkfit_vision PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/cpp/vision/include)
    # The ONNX Runtime C API headers are PRIVATE on purpose. Only onnx_detector.cpp
    # includes them, and that header is four hundred kilobytes: making it PUBLIC would
    # push it through every target that merely wants to know what a Detector is.
    #
    # Headers only. There is no import library and nothing to link against, because the
    # runtime is opened with LoadLibrary at startup, so the worker builds and runs on a
    # machine where ONNX Runtime is not installed.
    target_include_directories(parkfit_vision PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/cpp/third_party/onnxruntime)
    target_link_libraries(parkfit_vision PUBLIC parkfit_core)
    if(UNIX AND NOT APPLE)
      target_link_libraries(parkfit_vision PUBLIC ${CMAKE_DL_LIBS})
    endif()
  else()
    target_include_directories(parkfit_vision INTERFACE
      ${CMAKE_CURRENT_SOURCE_DIR}/cpp/vision/include)
    target_link_libraries(parkfit_vision INTERFACE parkfit_core)
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(PARKFIT_BUILD_TESTS)
  enable_testing()
  file(GLOB PARKFIT_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/test_*.cpp)
  foreach(test_src ${PARKFIT_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE parkfit_core)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests)
    if(PARKFIT_BUILD_VISION)
      target_link_libraries(${test_name} PRIVATE parkfit_vision)
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/apps/pf_cv_worker/main.cpp)
  add_executable(pf_cv_worker cpp/apps/pf_cv_worker/main.cpp)
  target_link_libraries(pf_cv_worker PRIVATE parkfit_core parkfit_vision)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/apps/pf_routecli/main.cpp)
  add_executable(pf_routecli cpp/apps/pf_routecli/main.cpp)
  target_link_libraries(pf_routecli PRIVATE parkfit_core)
endif()

# ---------------------------------------------------------------------------
# Python bindings (pybind11 is discovered from the active virtualenv)
# ---------------------------------------------------------------------------
if(PARKFIT_BUILD_BINDINGS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/bindings/module.cpp)
  find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
  if(Python3_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE PYBIND11_QUERY_RESULT)
    if(PYBIND11_QUERY_RESULT EQUAL 0)
      list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
      find_package(pybind11 CONFIG QUIET)
    endif()
  endif()

  if(pybind11_FOUND)
    pybind11_add_module(parkfit_native cpp/bindings/module.cpp)
    target_link_libraries(parkfit_native PRIVATE parkfit_core)
    if(PARKFIT_BUILD_VISION)
      target_link_libraries(parkfit_native PRIVATE parkfit_vision)
    endif()
    # Drop the .pyd straight into the package so `import parkfit_native` works from
    # an editable install without a separate copy step.
    set_target_properties(parkfit_native PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_SOURCE_DIR}/src/parkfit
      LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/src/parkfit
      LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/src/parkfit
      RUNTIME_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_SOURCE_DIR}/src/parkfit
      RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/src/parkfit
      RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/src/parkfit)
    message(STATUS "parkfit: Python bindings enabled")
  else()
    message(STATUS "parkfit: pybind11 not found, skipping Python bindings")
  endif()
endif()

message(STATUS "parkfit: build type       = ${CMAKE_BUILD_TYPE}")
message(STATUS "parkfit: compiler         = ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
