#include "gtest/gtest.h"
#include "image_preprocessor.h"
#include <vector>
#include <opencv2/opencv.hpp>


TEST(ImagePreprocessorTest, ProcessesImageCorrectly) {
    int model_input_size = 640;
    int pad_value = 114;
    float scale_factor = 1.0f / 255.0f;

    PromptDetector::ImagePreprocessor preprocessor(model_input_size, pad_value, scale_factor);

    int input_width = 400;
    int input_height = 300;
    cv::Mat input_image(input_height, input_width, CV_8UC3, cv::Scalar(255, 128, 64)); // B=255, G=128, R=64

    std::vector<float> output_tensor(3 * model_input_size * model_input_size);
    preprocessor.Process(input_image, output_tensor.data());
    float scale = std::min(static_cast<float>(model_input_size) / input_width, static_cast<float>(model_input_size) / input_height);
    int new_width = static_cast<int>(round(input_width * scale));
    int new_height = static_cast<int>(round(input_height * scale));
    int pad_top = (model_input_size - new_height) / 2;
    int pad_left = (model_input_size - new_width) / 2;
    int pad_pixel_index_r = 0;
    int pad_pixel_index_g = model_input_size * model_input_size;
    int pad_pixel_index_b = 2 * model_input_size * model_input_size;
    float expected_pad_val = pad_value * scale_factor;
    EXPECT_NEAR(output_tensor[pad_pixel_index_r], expected_pad_val, 1e-5);
    EXPECT_NEAR(output_tensor[pad_pixel_index_g], expected_pad_val, 1e-5);
    EXPECT_NEAR(output_tensor[pad_pixel_index_b], expected_pad_val, 1e-5);

    int center_x = pad_left + new_width / 2;
    int center_y = pad_top + new_height / 2;
    int center_pixel_index = center_y * model_input_size + center_x;

    int r_channel_offset = 0;
    int g_channel_offset = model_input_size * model_input_size;
    int b_channel_offset = 2 * model_input_size * model_input_size;

    EXPECT_NEAR(output_tensor[r_channel_offset + center_pixel_index], 64.0f * scale_factor, 1e-5);
    EXPECT_NEAR(output_tensor[g_channel_offset + center_pixel_index], 128.0f * scale_factor, 1e-5);
    EXPECT_NEAR(output_tensor[b_channel_offset + center_pixel_index], 255.0f * scale_factor, 1e-5);
}
