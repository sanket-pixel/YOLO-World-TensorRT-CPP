#ifndef PROMPT_DETECTOR_DATA_TYPES_H
#define PROMPT_DETECTOR_DATA_TYPES_H

#include <string>
#include <opencv2/opencv.hpp>

namespace PromptDetector {

struct Detection {
  cv::Rect bounding_box;
  float score;
  int class_id;
  std::string class_name;
};

} // namespace PromptDetector

#endif // PROMPT_DETECTOR_DATA_TYPES_H
