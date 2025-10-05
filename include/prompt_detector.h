#ifndef PROMPT_DETECTOR_H
#define PROMPT_DETECTOR_H

#include "data_types.h"
#include "detector_postprocessor.h"
#include "image_preprocessor.h"
#include "text_preprocessor.h"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace nvinfer1 {
    class IExecutionContext;
    class ICudaEngine;
    class IRuntime;
}

namespace PromptDetector {

class PromptDetector {
public:
    PromptDetector(const std::string& text_encoder_engine_path,
                   const std::string& detector_engine_path,
                   const std::string& postprocessor_onnx_path,
                   const std::string& vocab_path);

    ~PromptDetector();

    bool Load();
    bool SetVocabulary(const std::string& prompt);
    bool Detect(const cv::Mat& input_image, std::vector<Detection>& detections);

private:
    std::shared_ptr<nvinfer1::ICudaEngine> LoadEngineFromFile(const std::string& path);
    void AllocateGpuBuffers();
    void FreeGpuBuffers();

    std::string text_encoder_path_;
    std::string detector_path_;
    std::string postprocessor_onnx_path_;
    std::string vocab_path_;

    std::unique_ptr<ImagePreprocessor> image_preprocessor_;
    std::unique_ptr<TextPreprocessor> text_preprocessor_;
    std::unique_ptr<Postprocessor> postprocessor_;

    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> text_encoder_engine_;
    std::shared_ptr<nvinfer1::ICudaEngine> detector_engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> text_encoder_context_;
    std::shared_ptr<nvinfer1::IExecutionContext> detector_context_;

    Ort::Env ort_env_;
    Ort::SessionOptions ort_session_options_;
    std::unique_ptr<Ort::Session> postprocessor_session_;

    std::vector<std::string> ort_input_names_;
    std::vector<std::string> ort_output_names_;

    std::map<std::string, void*> gpu_buffers_;
};

} // namespace PromptDetector

#endif // PROMPT_DETECTOR_H

