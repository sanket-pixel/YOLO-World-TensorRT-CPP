#include "detector_postprocessor.h"

namespace PromptDetector {

Postprocessor::Postprocessor(float score_threshold) : score_threshold_(score_threshold) {}

void Postprocessor::Process(
    const std::vector<float>& bboxes,
    const std::vector<float>& scores,
    const std::vector<int32_t>& labels,
    const std::vector<bool>& nms_mask,
    std::vector<Detection>& detections) {

  detections.clear();
  const size_t num_detections = scores.size();

  for (size_t i = 0; i < num_detections; ++i) {
    if (!nms_mask[i] || scores[i] < score_threshold_) {
      continue;
    }

    Detection det;
    det.score = scores[i];
    det.class_id = labels[i];

    // Convert from (x1, y1, x2, y2) to (x, y, width, height).
    float x1 = bboxes[i * 4 + 0];
    float y1 = bboxes[i * 4 + 1];
    float x2 = bboxes[i * 4 + 2];
    float y2 = bboxes[i * 4 + 3];

    det.bounding_box.x = x1;
    det.bounding_box.y = y1;
    det.bounding_box.width = x2 - x1;
    det.bounding_box.height = y2 - y1;

    detections.push_back(det);
  }
}

}
