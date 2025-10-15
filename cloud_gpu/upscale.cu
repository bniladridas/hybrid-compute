#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <exception>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

__host__ __device__ float cubicInterpolate(float p0, float p1, float p2, float p3, float t) {
    return p1 + 0.5f * t * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + t * (3.0f * (p1 - p2) + p3 - p0));
}

__host__ __device__ float getBicubicValue(uchar* input, int in_w, int in_h, float gx, float gy, int c) {
    int gxi = (int)gx;
    int gyi = (int)gy;

    float vals[4][4];
    for (int m = -1; m <= 2; m++) {
        for (int n = -1; n <= 2; n++) {
            int px = min(max(gxi + m, 0), in_w - 1);
            int py = min(max(gyi + n, 0), in_h - 1);
            vals[m + 1][n + 1] = input[(py * in_w + px) * 3 + c];
        }
    }

    float col[4];
    for (int m = 0; m < 4; m++) {
        col[m] = cubicInterpolate(vals[m][0], vals[m][1], vals[m][2], vals[m][3], gx - gxi);
    }

    float value = cubicInterpolate(col[0], col[1], col[2], col[3], gy - gyi);

    return min(max(value, 0.0f), 255.0f);
}

__global__ void bicubicUpscaleKernel(uchar* input, uchar* output, int in_w, int in_h, int scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in_w || y >= in_h) return;

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < scale; i++) {
            for (int j = 0; j < scale; j++) {
                float gx = (float)x + (float)i / (float)scale;
                float gy = (float)y + (float)j / (float)scale;

                float value = getBicubicValue(input, in_w, in_h, gx, gy, c);

                int out_idx = ((y * scale + j) * (in_w * scale) + (x * scale + i)) * 3 + c;
                output[out_idx] = (uchar)value;
            }
        }
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            return -1; \
        } \
    } while (0)

int main(int argc, char** argv) {
    if (argc > 4) {
        std::cerr << "Usage: ./upscale [input_file] [output_file] [scale]\n";
        return -1;
    }
    std::string input_file = (argc > 1) ? argv[1] : "input_tile.jpg";
    std::string output_file = (argc > 2) ? argv[2] : "output_tile.jpg";
    int scale = 2;
    if (argc > 3) {
        try {
            scale = std::stoi(argv[3]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid scale value provided: " << argv[3] << std::endl;
            return -1;
        }
    }
    if (scale <= 1) {
        std::cerr << "Error: scale must be greater than 1\n";
        return -1;
    }

    // Load the input image
    cv::Mat input = cv::imread(input_file);
    if (input.empty()) {
        std::cerr << "Error: Could not load image from " << input_file << "!" << std::endl;
        return -1;
    }

    // Calculate dimensions
    int in_w = input.cols;
    int in_h = input.rows;
    int out_w = in_w * scale;
    int out_h = in_h * scale;

    // Create output image
    cv::Mat output(out_h, out_w, CV_8UC3);

    // Check for GPU availability
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    bool useGPU = deviceCount > 0;

    // Allocate memory
    uchar *d_input, *d_output;
    size_t input_size = in_w * in_h * 3 * sizeof(uchar);
    size_t output_size = out_w * out_h * 3 * sizeof(uchar);

    if (useGPU) {
        CUDA_CHECK(cudaMallocManaged(&d_input, input_size));
        CUDA_CHECK(cudaMallocManaged(&d_output, output_size));
        memcpy(d_input, input.data, input_size);
    } else {
        d_input = input.data;
        d_output = output.data;
    }

    if (useGPU) {
        // Launch kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((in_w + blockDim.x - 1) / blockDim.x, (in_h + blockDim.y - 1) / blockDim.y);
        bicubicUpscaleKernel<<<gridDim, blockDim>>>(d_input, d_output, in_w, in_h, scale);

        // Synchronize and check for kernel errors
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back to host
        memcpy(output.data, d_output, output_size);
    } else {
        // CPU fallback
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < in_h; y++) {
            for (int x = 0; x < in_w; x++) {
                for (int c = 0; c < 3; c++) {
                    for (int i = 0; i < scale; i++) {
                        for (int j = 0; j < scale; j++) {
                            float gx = (float)x + (float)i / (float)scale;
                            float gy = (float)y + (float)j / (float)scale;
                            float value = getBicubicValue(d_input, in_w, in_h, gx, gy, c);
                            int out_idx = ((y * scale + j) * out_w + (x * scale + i)) * 3 + c;
                            d_output[out_idx] = (uchar)value;
                        }
                    }
                }
            }
        }
    }

    // Free memory if GPU
    if (useGPU) {
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }

    // Save output image
    cv::imwrite(output_file, output);

    std::cout << "Upscaling complete. Output saved to " << output_file << std::endl;

    return 0;
}
