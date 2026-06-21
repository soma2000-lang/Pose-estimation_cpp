cmake_minimum_required(VERSION 3.20)
project(oop_detect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# ── OpenCV ───────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV ${OpenCV_VERSION} found: ${OpenCV_INCLUDE_DIRS}")

# ── ONNX Runtime ─────────────────────────────────────────────────────────────
# Expected install location: /usr/local/onnxruntime
# Install with: see tools/setup/install_onnxruntime.sh
set(ORT_ROOT   "/usr/local/onnxruntime"
    CACHE PATH "ONNX Runtime root directory")
set(ORT_INCLUDE "${ORT_ROOT}/include")
set(ORT_LIB     "${ORT_ROOT}/lib/libonnxruntime.so")

if(NOT EXISTS "${ORT_LIB}")
    message(FATAL_ERROR
        "ONNX Runtime library not found at ${ORT_LIB}\n"
        "Run: tools/setup/install_onnxruntime.sh")
endif()

# ── Include paths ────────────────────────────────────────────────────────────
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_INCLUDE}
)

# ── Sources ───────────────────────────────────────────────────────────────────
add_executable(oop_detect
    src/main.cpp
    src/detector_onnx.cpp
    src/pipeline.cpp
)

# ── Link ─────────────────────────────────────────────────────────────────────
target_link_libraries(oop_detect
    ${OpenCV_LIBS}
    ${ORT_LIB}
    stdc++fs          # std::filesystem on older compilers
)

# ── Optimisation flags ────────────────────────────────────────────────────────
target_compile_options(oop_detect PRIVATE
    -O3 -march=native -Wall -Wextra -Wpedantic
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "ORT root:   ${ORT_ROOT}")
