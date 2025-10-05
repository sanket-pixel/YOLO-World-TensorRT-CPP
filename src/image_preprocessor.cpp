#include "image_preprocessor.h"
#include <vector>

namespace PromptDetector {

ImagePreprocessor::ImagePreprocessor(int model_input_size, int pad_value, float scale_factor)
    : model_input_size_(model_input_size), pad_value_(pad_value), scale_factor_(scale_factor) {}

void ImagePreprocessor::Process(const cv::Mat& input_image, float* output_tensor) {
    cv::Mat resized_image = ResizeKeepAspectRatio(input_image, model_input_size_, scale_);
    cv::Mat padded_image = Letterbox(resized_image, model_input_size_, pad_top_, pad_left_);

    cv::Mat float_image;
    padded_image.convertTo(float_image, CV_32FC3, scale_factor_);

    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);

    int channel_size = model_input_size_ * model_input_size_;

    memcpy(output_tensor, channels[2].data, channel_size * sizeof(float));
    memcpy(output_tensor + channel_size, channels[1].data, channel_size * sizeof(float));
    memcpy(output_tensor + 2 * channel_size, channels[0].data, channel_size * sizeof(float));
}

cv::Mat ImagePreprocessor::ResizeKeepAspectRatio(const cv::Mat& input_image, int target_size, float& scale) {
    int original_width = input_image.cols;
    int original_height = input_image.rows;

    scale = std::min(static_cast<float>(target_size) / original_width, static_cast<float>(target_size) / original_height);

    int new_width = static_cast<int>(round(original_width * scale));
    int new_height = static_cast<int>(round(original_height * scale));

    cv::Mat resized_image;
    cv::resize(input_image, resized_image, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    return resized_image;
}

cv::Mat ImagePreprocessor::Letterbox(const cv::Mat& input_image, int target_size, int& pad_top, int& pad_left) {
    int new_height = input_image.rows;
    int new_width = input_image.cols;

    pad_top = (target_size - new_height) / 2;
    int pad_bottom = target_size - new_height - pad_top;
    pad_left = (target_size - new_width) / 2;
    int pad_right = target_size - new_width - pad_left;

    cv::Mat padded_image;
    cv::copyMakeBorder(input_image, padded_image, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT,
                       cv::Scalar(pad_value_, pad_value_, pad_value_));
    return padded_image;
}

} // namespace PromptDetector

