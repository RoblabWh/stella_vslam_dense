cmake_minimum_required(VERSION 3.5)
project(InstaStitcherDirty)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(nlohmann_json REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(insta360_stitcher SHARED
  src/insta360_stitcher.cpp
  src/utils/geometry.cpp
  src/camera_models/camera_model.cpp
  src/camera_models/double_sphere.cpp
  src/camera_models/extended_unified.cpp
)
target_include_directories(insta360_stitcher PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(insta360_stitcher
  PUBLIC
  opencv_core opencv_imgproc opencv_imgcodecs
  nlohmann_json::nlohmann_json
  yaml-cpp
)

install(TARGETS insta360_stitcher)

