#ifndef PROMPT_DETECTOR_POSTPROCESSOR_H
#define PROMPT_DETECTOR_POSTPROCESSOR_H

#include "data_types.h"
#include <vector>
#include <cstdint>

namespace PromptDetector {

class Postprocessor {
public:
  explicit Postprocessor(float score_threshold);

  void Process(
      const std::vector<float>& bboxes,
      const std::vector<float>& scores,
      const std::vector<int32_t>& labels,
      const std::vector<bool>& nms_mask,
      std::vector<Detection>& detections
  );

private:
  float score_threshold_;
};

}

#endif