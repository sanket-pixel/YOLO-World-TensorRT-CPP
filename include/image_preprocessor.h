#ifndef PROMPT_DETECTOR_IMAGE_PREPROCESSOR_H
#define PROMPT_DETECTOR_IMAGE_PREPROCESSOR_H

#include <opencv2/opencv.hpp>

namespace PromptDetector {

class ImagePreprocessor {
public:
  ImagePreprocessor(int model_input_size, int pad_value, float scale_factor);
  void Process(const cv::Mat& input_image, float* output_tensor);

private:
  cv::Mat ResizeKeepAspectRatio(const cv::Mat& input_image, int target_size, float& scale);
  cv::Mat Letterbox(const cv::Mat& input_image, int target_size, int& pad_top, int& pad_left);

  int model_input_size_;
  int pad_value_;
  float scale_factor_;
  float scale_{1.0f};
  int pad_top_{0};
  int pad_left_{0};
};

} // namespace PromptDetector

#endif // PROMPT_DETECTOR_IMAGE_PREPROCESSOR_H
