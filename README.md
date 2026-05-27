# CUDA-Accelerated Batch Image Processing Pipeline

## Project Overview

This project implements a GPU-accelerated batch image processing system using CUDA and OpenCV CUDA modules. The application processes a large collection of images in parallel on the GPU to improve execution speed and demonstrate high-performance image processing using CUDA.

The system applies multiple image processing operations, including:
- Grayscale Conversion
- Gaussian Blur
- Sobel Edge Detection
- Thresholding

The project demonstrates the use of GPU computing for accelerating image processing workloads on large datasets.

## Features
- GPU-accelerated image processing using CUDA
- Batch processing of hundreds of images
- CUDA-based edge detection pipeline
- Parallel image transformations
- Benchmarking and execution timing
- Input and output image visualization
- OpenCV CUDA integration

## Technologies Used
- CUDA
- C++ (C++17)
- OpenCV CUDA
- NVIDIA GPU
- CUDA Runtime API

## CUDA Concepts Demonstrated
This project demonstrates several CUDA concepts, including:
- GPU Memory Allocation
- CUDA Streams
- Parallel Kernel Execution
- Device-to-Host and Host-to-Device Transfers
- GPU-Accelerated Filtering
- CUDA Event Timing

## Processing Pipeline
The image processing pipeline follows these steps:
1. Load image from disk
2. Upload image to GPU memory
3. Convert image to grayscale
4. Apply Gaussian blur
5. Apply Sobel edge detection
6. Apply thresholding
7. Download processed image
8. Save output image to disk

## Project Structure
```text
project/
│
├── input_images/
├── output_images/
├── screenshots/
├── src/
│   └── main.cu
│
├── benchmark_results.txt
├── execution_log.txt
└── README.md
```

## Build Instructions

### Requirements
- CUDA Toolkit
- OpenCV with CUDA support
- NVIDIA GPU
- C++17 compatible compiler

### Build
To build the project on a Linux environment (or compatible bash environment), run:
```bash
nvcc src/main.cu -o image_processor `pkg-config --cflags --libs opencv4` -std=c++17
```

*(Note for Windows users: You may want to integrate with CMake or MSBuild depending on your OpenCV installation.)*

## Run Instructions
```bash
./image_processor
```

The program automatically processes all images inside the `input_images` directory and stores processed outputs in the `output_images` directory.

## Benchmark Results
Example benchmark output stored in `benchmark_results.txt`:
```
Processed 100 images in 1.24 seconds
Average GPU processing time per image: 0.0124 seconds
```

## Educational Goals
This project demonstrates how CUDA can accelerate large-scale image processing workloads through GPU parallelism and optimized memory transfers. The implementation highlights the advantages of GPU computing for computer vision and signal/image processing applications.

---
**Author:** Shane
