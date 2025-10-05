#include "prompt_detector.h"
#include "data_types.h"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cxxopts.hpp> // For argument parsing

void DrawDetections(cv::Mat& image, const std::vector<PromptDetector::Detection>& detections, const std::string& class_name) {
    for (const auto& det : detections) {
        cv::rectangle(image, det.bounding_box, cv::Scalar(0, 255, 0), 2);

        std::string label = class_name + ": " + cv::format("%.2f", det.score);
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int top = std::max(det.bounding_box.y, label_size.height);
        cv::rectangle(image, cv::Point(det.bounding_box.x, top - label_size.height),
                      cv::Point(det.bounding_box.x + label_size.width, top + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(image, label, cv::Point(det.bounding_box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

int main(int argc, char* argv[]) {
    cxxopts::Options options("PromptDetectorApp", "Open Vocabulary Object Detection with TensorRT");

    options.add_options()
        ("text_encoder", "Path to the text encoder engine", cxxopts::value<std::string>())
        ("detector", "Path to the detector engine", cxxopts::value<std::string>())
        ("postprocessor", "Path to the postprocessor engine", cxxopts::value<std::string>())
        ("vocab", "Path to the vocab.json file", cxxopts::value<std::string>())
        ("image", "Path to the input image", cxxopts::value<std::string>())
        ("prompt", "Detection prompt (e.g., 'a person walking a dog')", cxxopts::value<std::string>())
        ("output", "Path to save the output image", cxxopts::value<std::string>()->default_value("output.jpg"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    const std::vector<std::string> required_args = {"text_encoder", "detector", "postprocessor", "vocab", "image", "prompt"};
    for (const auto& arg : required_args) {
        if (result.count(arg) == 0) {
            std::cerr << "Error: Missing required argument: --" << arg << std::endl;
            std::cout << options.help() << std::endl;
            return 1;
        }
    }

    try {
        PromptDetector::PromptDetector detector(
            result["text_encoder"].as<std::string>(),
            result["detector"].as<std::string>(),
            result["postprocessor"].as<std::string>(),
            result["vocab"].as<std::string>()
        );

        std::cout << "Loading models..." << std::endl;
        if (!detector.Load()) {
            std::cerr << "Error: Failed to load TensorRT engines." << std::endl;
            return 1;
        }
        std::string prompt = result["prompt"].as<std::string>();
        std::cout << "Setting vocabulary with prompt: \"" << prompt << "\"" << std::endl;
        if (!detector.SetVocabulary(prompt)) {
            std::cerr << "Error: Failed to set vocabulary." << std::endl;
            return 1;
        }

        std::string image_path = result["image"].as<std::string>();
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Could not read image from path: " << image_path << std::endl;
            return 1;
        }

        std::cout << "Running detection..." << std::endl;
        std::vector<PromptDetector::Detection> detections;
        if (!detector.Detect(image, detections)) {
            std::cerr << "Error: Failed to run detection." << std::endl;
            return 1;
        }
        std::cout << "Found " << detections.size() << " detections." << std::endl;
        for (const auto& det : detections) {
            std::cout << "  - Class: " << det.class_id
                      << ", Score: " << det.score
                      << ", Box: [x=" << det.bounding_box.x << ", y=" << det.bounding_box.y
                      << ", w=" << det.bounding_box.width << ", h=" << det.bounding_box.height << "]" << std::endl;
        }
        DrawDetections(image, detections, prompt);
        std::string output_path = result["output"].as<std::string>();
        if (cv::imwrite(output_path, image)) {
            std::cout << "Saved output image with detections to: " << output_path << std::endl;
        } else {
            std::cerr << "Error: Failed to save output image to: " << output_path << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
