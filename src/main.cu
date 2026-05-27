#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <ctime>

// Include OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

namespace fs = std::filesystem;

void log_execution(const std::string& message) {
    std::ofstream log_file("execution_log.txt", std::ios_base::app);
    if (log_file.is_open()) {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::string time_str = std::ctime(&now_c);
        time_str.pop_back(); // Remove trailing newline from ctime
        log_file << "[" << time_str << "] " << message << std::endl;
    }
}

int main() {
    std::string input_dir = "input_images";
    std::string output_dir = "output_images";

    log_execution("Starting CUDA batch image processing pipeline.");
    
    // Check if CUDA is available
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    if (cuda_devices == 0) {
        std::cerr << "No CUDA devices found! Ensure OpenCV is built with CUDA support." << std::endl;
        log_execution("Error: No CUDA devices found.");
        return -1;
    }
    
    cv::cuda::setDevice(0);
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    // Gather input images
    std::vector<std::string> image_paths;
    if (fs::exists(input_dir)) {
        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                // Simple extension check
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    image_paths.push_back(entry.path().string());
                }
            }
        }
    } else {
        std::cerr << "Input directory '" << input_dir << "' does not exist." << std::endl;
        return -1;
    }

    if (image_paths.empty()) {
        std::cout << "No images found in " << input_dir << " directory." << std::endl;
        log_execution("No images found to process.");
        return 0;
    }

    int processed_count = 0;
    
    // Create output directory if it doesn't exist (though we created it in setup)
    if (!fs::exists(output_dir)) {
        fs::create_directory(output_dir);
    }

    // GPU Warm-up step
    cv::Mat dummy = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::cuda::GpuMat d_dummy;
    d_dummy.upload(dummy);

    std::cout << "Processing " << image_paths.size() << " images..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto& path : image_paths) {
        fs::path p(path);
        std::string filename = p.filename().string();
        
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            log_execution("Failed to read image: " + filename);
            continue;
        }

        cv::cuda::GpuMat d_img, d_gray, d_blurred, d_sobel_32f, d_sobel, d_thresholded;
        
        // 1. Upload image to GPU memory
        d_img.upload(img);
        
        // 2. Convert image to grayscale
        cv::cuda::cvtColor(d_img, d_gray, cv::COLOR_BGR2GRAY);
        
        // 3. Apply Gaussian blur
        auto gaussian_filter = cv::cuda::createGaussianFilter(d_gray.type(), d_gray.type(), cv::Size(5, 5), 1.5);
        gaussian_filter->apply(d_gray, d_blurred);
        
        // 4. Apply Sobel edge detection
        auto sobel_filter = cv::cuda::createSobelFilter(d_blurred.type(), CV_32F, 1, 1);
        sobel_filter->apply(d_blurred, d_sobel_32f);
        
        // Convert scale back to 8-bit
        cv::cuda::absSum(d_sobel_32f, d_sobel_32f); 
        d_sobel_32f.convertTo(d_sobel, CV_8U);
        
        // 5. Apply thresholding
        cv::cuda::threshold(d_sobel, d_thresholded, 50.0, 255.0, cv::THRESH_BINARY);
        
        // 6. Download processed image from GPU memory
        cv::Mat result;
        d_thresholded.download(result);
        
        // 7. Save output image to disk
        std::string output_path = (fs::path(output_dir) / filename).string();
        cv::imwrite(output_path, result);
        
        processed_count++;
        log_execution("Processed image: " + filename);
    }

    // Benchmark calculation
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    double avg_time = processed_count > 0 ? duration.count() / processed_count : 0.0;

    std::cout << "\n--- Benchmark Results ---" << std::endl;
    std::cout << "Processed " << processed_count << " images in " << duration.count() << " seconds" << std::endl;
    std::cout << "Average GPU processing time per image: " << avg_time << " seconds" << std::endl;

    // Log benchmark to file
    std::ofstream bench_file("benchmark_results.txt", std::ios_base::app);
    if (bench_file.is_open()) {
        bench_file << "Processed " << processed_count << " images in " << duration.count() << " seconds\n";
        bench_file << "Average GPU processing time per image: " << avg_time << " seconds\n";
        bench_file << "--------------------------------------------------\n";
    }
    
    log_execution("Pipeline finished successfully. Processed " + std::to_string(processed_count) + " images.");
    return 0;
}
