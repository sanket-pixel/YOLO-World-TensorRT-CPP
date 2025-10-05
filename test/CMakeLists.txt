cmake_minimum_required(VERSION 3.18)
project(PromptDetectorTests)

# Fetch GoogleTest framework
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

find_package(OpenCV REQUIRED)

# Add the test executable
add_executable(
        run_tests
        src/test_image_preprocessor.cpp
        src/test_text_preprocessor.cpp
)

# Link the test executable against GoogleTest AND our project's library.
# Our library (prompt_detector_lib) already contains the compiled image_preprocessor.cpp
# and is already linked to OpenCV, so we get everything we need.
target_link_libraries(run_tests
        PRIVATE
        gtest_main
        prompt_detector_lib
        ${OpenCV_LIBS}
)

include(GoogleTest)
gtest_discover_tests(run_tests)
