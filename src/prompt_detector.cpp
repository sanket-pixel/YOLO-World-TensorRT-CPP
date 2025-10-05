#include "prompt_detector.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) \
                      << "\"" << std::endl;                              \
            return false;                                                \
        }                                                                \
    } while (0)

namespace {
// A simple logger for TensorRT
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};
}

namespace PromptDetector {

PromptDetector::PromptDetector(const std::string& text_encoder_engine_path,
                             const std::string& detector_engine_path,
                             const std::string& postprocessor_onnx_path,
                             const std::string& vocab_path)
    : text_encoder_path_(text_encoder_engine_path),
      detector_path_(detector_engine_path),
      postprocessor_onnx_path_(postprocessor_onnx_path),
      vocab_path_(vocab_path),
      ort_env_(ORT_LOGGING_LEVEL_WARNING, "PromptDetectorORT") {

    image_preprocessor_ = std::make_unique<ImagePreprocessor>(640, 114, 0.00392157);
    text_preprocessor_ = std::make_unique<TextPreprocessor>(vocab_path_);
    postprocessor_ = std::make_unique<Postprocessor>(0.15);
}

PromptDetector::~PromptDetector() {
    FreeGpuBuffers();
}

std::shared_ptr<nvinfer1::ICudaEngine> PromptDetector::LoadEngineFromFile(const std::string& path) {
    std::ifstream engine_file(path, std::ios::binary);
    if (!engine_file.is_open()) {
        std::cerr << "Error opening engine file: " << path << std::endl;
        return nullptr;
    }

    engine_file.seekg(0, std::ios::end);
    long int fsize = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(fsize);
    engine_file.read(engine_data.data(), fsize);

    return std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(engine_data.data(), fsize)
    );
}

bool PromptDetector::Load() {
    static TrtLogger logger;
    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime_) return false;

    text_encoder_engine_ = LoadEngineFromFile(text_encoder_path_);
    detector_engine_ = LoadEngineFromFile(detector_path_);
    if (!text_encoder_engine_ || !detector_engine_) return false;

    text_encoder_context_ = std::shared_ptr<nvinfer1::IExecutionContext>(text_encoder_engine_->createExecutionContext());
    detector_context_ = std::shared_ptr<nvinfer1::IExecutionContext>(detector_engine_->createExecutionContext());
    if (!text_encoder_context_ || !detector_context_) return false;

    ort_session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    postprocessor_session_ = std::make_unique<Ort::Session>(ort_env_, postprocessor_onnx_path_.c_str(), ort_session_options_);

    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < postprocessor_session_->GetInputCount(); ++i) {
        auto allocated_name = postprocessor_session_->GetInputNameAllocated(i, allocator);
        ort_input_names_.push_back(allocated_name.get());
    }

    for (size_t i = 0; i < postprocessor_session_->GetOutputCount(); ++i) {
        auto allocated_name = postprocessor_session_->GetOutputNameAllocated(i, allocator);
        ort_output_names_.push_back(allocated_name.get());
    }

    AllocateGpuBuffers();
    return true;
}

void PromptDetector::AllocateGpuBuffers() {
    auto allocate_for_engine = [&](const std::shared_ptr<nvinfer1::ICudaEngine>& engine) {
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            auto dims = engine->getBindingDimensions(i);
            size_t size = 1;
            for (int j = 0; j < dims.nbDims; ++j) size *= dims.d[j];
            cudaMalloc(&gpu_buffers_[engine->getBindingName(i)], size * sizeof(float));
        }
    };
    allocate_for_engine(text_encoder_engine_);
    allocate_for_engine(detector_engine_);
}

void PromptDetector::FreeGpuBuffers() {
    for (auto& pair : gpu_buffers_) {
        cudaFree(pair.second);
    }
    gpu_buffers_.clear();
}

bool PromptDetector::SetVocabulary(const std::string& prompt) {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> attention_mask;
    text_preprocessor_->Process(prompt, token_ids, attention_mask);

    CHECK_CUDA(cudaMemcpy(gpu_buffers_["text_prompt_tokens"], token_ids.data(), token_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_buffers_["text_prompt_padding_mask"], attention_mask.data(), attention_mask.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    int num_bindings = text_encoder_engine_->getNbBindings();
    std::vector<void*> bindings(num_bindings);
    for(int i = 0; i < num_bindings; ++i) {
        bindings[i] = gpu_buffers_[text_encoder_engine_->getBindingName(i)];
    }

    return text_encoder_context_->executeV2(bindings.data());
}

bool PromptDetector::Detect(const cv::Mat& input_image, std::vector<Detection>& detections) {
    std::vector<float> image_data(1 * 3 * 640 * 640);
    image_preprocessor_->Process(input_image, image_data.data());
    CHECK_CUDA(cudaMemcpy(gpu_buffers_["image"], image_data.data(), image_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    float width = static_cast<float>(input_image.cols);
    float height = static_cast<float>(input_image.rows);
    CHECK_CUDA(cudaMemcpy(gpu_buffers_["original_image_width"], &width, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_buffers_["original_image_height"], &height, sizeof(float), cudaMemcpyHostToDevice));

    int num_detector_bindings = detector_engine_->getNbBindings();
    std::vector<void*> detector_bindings(num_detector_bindings);
    for (int i = 0; i < num_detector_bindings; ++i) {
        detector_bindings[i] = gpu_buffers_[detector_engine_->getBindingName(i)];
    }
    if (!detector_context_->executeV2(detector_bindings.data())) return false;

    std::vector<float> onnx_bboxes(8400 * 4);
    std::vector<float> onnx_scores(8400 * 2);
    CHECK_CUDA(cudaMemcpy(onnx_bboxes.data(), gpu_buffers_["onnx_bbox_xyxy"], onnx_bboxes.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(onnx_scores.data(), gpu_buffers_["onnx_confidence"], onnx_scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cudaStreamSynchronize(0);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> onnx_bboxes_shape = {8400, 4};
    std::vector<int64_t> onnx_scores_shape = {8400, 2};
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, onnx_bboxes.data(), onnx_bboxes.size(), onnx_bboxes_shape.data(), onnx_bboxes_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, onnx_scores.data(), onnx_scores.size(), onnx_scores_shape.data(), onnx_scores_shape.size()));

    std::vector<const char*> input_names_ptr;
    for (const auto& name : ort_input_names_) {
        input_names_ptr.push_back(name.c_str());
    }
    std::vector<const char*> output_names_ptr;
    for (const auto& name : ort_output_names_) {
        output_names_ptr.push_back(name.c_str());
    }

    auto output_tensors = postprocessor_session_->Run(Ort::RunOptions{nullptr}, input_names_ptr.data(), input_tensors.data(), input_tensors.size(), output_names_ptr.data(), output_names_ptr.size());

    float* bboxes_ptr = output_tensors[0].GetTensorMutableData<float>();
    float* scores_ptr = output_tensors[1].GetTensorMutableData<float>();
    int32_t* labels_ptr = output_tensors[2].GetTensorMutableData<int32_t>();
    bool* mask_ptr = output_tensors[3].GetTensorMutableData<bool>();

    size_t num_proposals = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[0];
    std::vector<float> bboxes(bboxes_ptr, bboxes_ptr + num_proposals * 4);
    std::vector<float> scores(scores_ptr, scores_ptr + num_proposals);
    std::vector<int32_t> labels(labels_ptr, labels_ptr + num_proposals);
    std::vector<bool> nms_mask(mask_ptr, mask_ptr + num_proposals);

    postprocessor_->Process(bboxes, scores, labels, nms_mask, detections);

    return true;
}

} // namespace PromptDetector

