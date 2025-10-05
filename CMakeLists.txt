cmake_minimum_required(VERSION 3.18)
project(PromptDetector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG v3.1.1
)
FetchContent_MakeAvailable(cxxopts)

FetchContent_Declare(
        httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG v0.15.3
)
FetchContent_MakeAvailable(httplib)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

if(NOT DEFINED TENSORRT_DIR)
    message(FATAL_ERROR "TensorRT not found! Please specify its location with -DTENSORRT_DIR=/path/to/tensorrt")
endif()

set(TENSORRT_INCLUDE_DIRS ${TENSORRT_DIR}/include)
set(TENSORRT_LIBRARY_DIRS ${TENSORRT_DIR}/lib)

find_library(NVINFER_LIBRARY nvinfer HINTS ${TENSORRT_LIBRARY_DIRS} REQUIRED)
find_library(NVONNXPARSER_LIBRARY nvonnxparser HINTS ${TENSORRT_LIBRARY_DIRS} REQUIRED)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin HINTS ${TENSORRT_LIBRARY_DIRS} REQUIRED)

set(ONNXRUNTIME_DIR "/home/sshah/libs/onnxruntime-linux-x64-gpu-1.17.3" CACHE PATH "Root of ONNX Runtime install")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h HINTS ${ONNXRUNTIME_DIR} PATH_SUFFIXES include REQUIRED)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_DIR} PATH_SUFFIXES lib REQUIRED)

message(STATUS "Found ONNX Runtime Include Dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Found ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")

include_directories(include)

add_library(
        prompt_detector_lib
        src/image_preprocessor.cpp
        src/text_preprocessor.cpp
        src/tokenizer.cpp
        src/prompt_detector.cpp
        src/detector_postprocessor.cpp
)

target_include_directories(prompt_detector_lib
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${TENSORRT_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(prompt_detector_lib
        PUBLIC
        nlohmann_json::nlohmann_json
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${NVINFER_LIBRARY}
        ${NVONNXPARSER_LIBRARY}
        ${NVINFER_PLUGIN_LIBRARY}
        ${ONNXRUNTIME_LIBRARY}
)

add_executable(
        prompt_detector_app
        src/main.cpp
        src/server.cpp
)

target_link_libraries(prompt_detector_app
        PUBLIC
        prompt_detector_lib
        cxxopts::cxxopts
)

add_executable(
        prompt_detector_server
        src/server.cpp
)

target_include_directories(prompt_detector_server PRIVATE ${httplib_SOURCE_DIR})

target_link_libraries(prompt_detector_server
        PUBLIC
        prompt_detector_lib
        httplib::httplib
        cxxopts::cxxopts
)

enable_testing()
add_subdirectory(test)

