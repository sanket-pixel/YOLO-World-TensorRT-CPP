#include "httplib.h"
#include "prompt_detector.h"
#include "nlohmann/json.hpp"
#include <opencv2/opencv.hpp>
#include "cxxopts.hpp"
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) \
                      << "\"" << std::endl;                              \
            /* In a server context, we might not want to exit */         \
        }                                                                \
    } while (0)


int main(int argc, char** argv) {
    cxxopts::Options options("PromptDetectorServer", "A C++ web server for the YOLO-World prompt detector.");
    options.add_options()
        ("t,text_encoder", "Path to text encoder engine", cxxopts::value<std::string>())
        ("d,detector", "Path to detector engine file", cxxopts::value<std::string>())
        ("p,postprocessor", "Path to postprocessor engine file", cxxopts::value<std::string>())
        ("v,vocab", "Path to vocabulary file", cxxopts::value<std::string>())
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (!result.count("text_encoder") || !result.count("detector") || !result.count("postprocessor") || !result.count("vocab")) {
        std::cerr << "Error: Missing required arguments. Use --help for usage." << std::endl;
        return 1;
    }

    auto detector = std::make_shared<PromptDetector::PromptDetector>(
        result["text_encoder"].as<std::string>(),
        result["detector"].as<std::string>(),
        result["postprocessor"].as<std::string>(),
        result["vocab"].as<std::string>()
    );

    if (!detector->Load()) {
        std::cerr << "Failed to load the detector model." << std::endl;
        return 1;
    }
    std::cout << "Detector loaded successfully." << std::endl;

    httplib::Server svr;

    svr.Options("/detect", [](const httplib::Request&, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.set_header("Access-Control-Allow-Methods", "POST");
        res.status = 204; // No Content
    });

    svr.Post("/detect", [&](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        std::cout << "--- New Request Received ---" << std::endl;
        std::cout << "Parameters found: " << req.params.size() << std::endl;
        for(const auto& param : req.params) {
            std::cout << "  - Param: '" << param.first << "' = '" << param.second << "'" << std::endl;
        }
        std::cout << "Files found: " << req.files.size() << std::endl;
        for(const auto& file : req.files) {
            std::cout << "  - File field: '" << file.first << "', Filename: '" << file.second.filename << "', Size: " << file.second.content.length() << " bytes" << std::endl;
        }
        std::cout << "--------------------------" << std::endl;

        if (!req.has_file("image") || !req.has_param("prompt")) {
            res.status = 400;
            res.set_content(R"({"error": "Missing image or prompt."})", "application/json");
            return;
        }

        const auto& image_file = req.get_file_value("image");
        const auto& prompt = req.get_param_value("prompt");

        std::vector<char> image_data(image_file.content.begin(), image_file.content.end());
        cv::Mat image = cv::imdecode(cv::Mat(image_data), cv::IMREAD_COLOR);

        if (image.empty()) {
            res.status = 400;
            res.set_content(R"({"error": "Failed to decode image."})", "application/json");
            return;
        }

        std::cout << "Received prompt: '" << prompt << "'" << std::endl;

        if (!detector->SetVocabulary(prompt)) {
             res.status = 500;
             res.set_content(R"({"error": "Failed to set vocabulary."})", "application/json");
             return;
        }

        std::vector<PromptDetector::Detection> detections;
        if (!detector->Detect(image, detections)) {
            res.status = 500;
            res.set_content(R"({"error": "Inference failed."})", "application/json");
            return;
        }

        nlohmann::json json_response = nlohmann::json::array();
        for (const auto& det : detections) {
            json_response.push_back({
                {"box", {det.bounding_box.x, det.bounding_box.y, det.bounding_box.width, det.bounding_box.height}},
                {"confidence", det.score},
                {"label", det.class_id}
            });
        }

        res.set_content(json_response.dump(), "application/json");
    });

    std::cout << "Server listening at http://localhost:8080" << std::endl;
    svr.listen("0.0.0.0", 8080);

    return 0;
}

