# YOLO-World Open Vocabulary Object Detection (C++ and TensorRT)

This project is a complete from-scratch implementation of open-vocabulary object detection using **YOLO-World**, written entirely in **C++** with **TensorRT** for inference and **ONNX Runtime** for postprocessing. It demonstrates how to run text-guided object detection without relying on Python, and includes a lightweight web interface to visualize detections in real time.


## Demo

A simple web interface is provided for demonstration. It allows users to select an image and enter a text prompt, then runs detection via the backend server.

**Demo Placeholder:**

![Demo](demo.gif)


---

---

## Project Overview

The system takes an image and a text prompt (for example, “dog” or “person with a backpack”) and detects the corresponding objects, even if those categories were not part of the original YOLO training set.
The entire inference pipeline runs natively in C++ and uses TensorRT for high-performance execution on NVIDIA GPUs.

---

## Core Components

1. **Text Encoder (TensorRT Engine)**
   The text prompt is tokenized and preprocessed entirely in C++. A custom tokenizer and `TextPreprocessor` handle vocabulary loading, token ID creation, and attention mask generation. The processed text is passed through a TensorRT engine for the YOLO-World text encoder to produce an embedding.

2. **Image Preprocessor**
   The input image is resized, normalized, and converted to CHW layout using OpenCV. This matches YOLO-World’s preprocessing pipeline but implemented from scratch in C++.

3. **Detector (TensorRT Engine)**
   The image tensor and the text embedding are fed to the YOLO-World detector engine, which outputs candidate bounding boxes and confidence scores.

4. **Postprocessor (ONNX Runtime)**
   The raw detector outputs are passed through an ONNX-based postprocessing model. This handles NMS (non-maximum suppression) and filtering. The results are converted into structured detections with bounding boxes, labels, and confidence scores.

5. **Web Interface**
   A simple web server and front-end interface allow users to upload an image and enter a text prompt. The C++ backend runs the entire inference pipeline and returns results to the browser for visualization.

---

## End-to-End Flow

Below is a simplified flowchart of the detection pipeline:

```
       +-------------------+
       |   Text Prompt     |
       +-------------------+
                  |
                  v
       +-------------------+
       |  C++ Tokenizer &  |
       | Text Preprocessor |
       +-------------------+
                  |
                  v
       +-------------------+
       | TensorRT Text     |
       | Encoder Engine    |
       +-------------------+
                  |
                  v
  +----------------------------+
  |  YOLO-World Detector (TRT) |
  |   Image + Text Embedding   |
  +----------------------------+
                  |
                  v
       +-------------------+
       | ONNX Postprocessor|
       +-------------------+
                  |
                  v
       +-------------------+
       |  Detections (CPU) |
       +-------------------+
                  |
                  v
       +-------------------+
       | Web Visualization |
       +-------------------+
```

---

## Technical Highlights

* Built entirely in **C++17** with no Python dependencies.
* **TensorRT** used for text encoder and detector engines.
* **ONNX Runtime** used for CPU postprocessing (NMS and filtering).
* **CUDA memory management** implemented manually for each binding.
* **Custom tokenizer and preprocessing** for text and image inputs.
* Designed to be modular, with clean separation between components:

    * `image_preprocessor.cpp`
    * `text_preprocessor.cpp`
    * `tokenizer.cpp`
    * `detector_postprocessor.cpp`
    * `prompt_detector.cpp` (main pipeline)

---


## Current Status

Model weights and engine files are not yet included in the repository.
Support for building and running the full pipeline will be added soon along with setup instructions.

---

## Future Work

* Add instructions for generating TensorRT engines from YOLO-World ONNX models.
* Add example scripts for benchmarking and profiling.
* Extend the web interface for multiple prompts and real-time camera input.

---

Would you like me to add a **"Build and Run (coming soon)"** section as a placeholder as well, so it looks complete on GitHub even before weights are added?
