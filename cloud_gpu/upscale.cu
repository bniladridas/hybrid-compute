/**
 * CUDA program for bicubic upscaling of images using GPU acceleration.
 * This program takes an input image tile, upscales it by a factor of 2 using bicubic interpolation,
 * and saves the result to an output file.
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

/**
 * Device function for cubic interpolation.
 * Interpolates between four points using the cubic Hermite spline.
 * @param p0, p1, p2, p3: The four control points
 * @param t: The interpolation parameter (0 <= t <= 1)
 * @return The interpolated value
 */
__device__ float cubicInterpolate(float p0, float p1, float p2, float p3, float t) {
    return p1 + 0.5f * t * (p2 - p0 + t * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + t * (3.0f * (p1 - p2) + p3 - p0)));
}

/**
 * CUDA kernel for bicubic upscaling.
 * Each thread processes one pixel of the input image and generates a scale x scale block in the output.
 * @param input: Pointer to input image data (RGB, row-major)
 * @param output: Pointer to output image data (RGB, row-major)
 * @param in_w, in_h: Input image dimensions
 * @param scale: Upscaling factor
 */
__global__ void bicubicUpscaleKernel(uchar* input, uchar* output, int in_w, int in_h, int scale) {
    // Calculate the pixel coordinates for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (x >= in_w || y >= in_h) return;

    // Process each color channel
    for (int c = 0; c < 3; c++) {
        // For each output pixel in the scale x scale block
        for (int i = 0; i < scale; i++) {
            for (int j = 0; j < scale; j++) {
                // Calculate the sub-pixel coordinates in the input image
                float gx = (float)x + (float)i / (float)scale;
                float gy = (float)y + (float)j / (float)scale;

                int gxi = (int)gx;
                int gyi = (int)gy;

                // Sample the 4x4 neighborhood for bicubic interpolation
                float vals[4][4];
                for (int m = -1; m <= 2; m++) {
                    for (int n = -1; n <= 2; n++) {
                        // Clamp coordinates to image bounds
                        int px = min(max(gxi + m, 0), in_w - 1);
                        int py = min(max(gyi + n, 0), in_h - 1);
                        vals[m + 1][n + 1] = input[(py * in_w + px) * 3 + c];
                    }
                }

                // Interpolate along x for each row
                float col[4];
                for (int m = 0; m < 4; m++) {
                    col[m] = cubicInterpolate(vals[m][0], vals[m][1], vals[m][2], vals[m][3], gx - gxi);
                }

                // Interpolate along y
                float value = cubicInterpolate(col[0], col[1], col[2], col[3], gy - gyi);

                // Calculate output index and clamp value
                int out_idx = ((y * scale + j) * (in_w * scale) + (x * scale + i)) * 3 + c;
                output[out_idx] = min(max((int)value, 0), 255);
            }
        }
    }
}

/**
 * Macro for checking CUDA API calls and handling errors.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            return -1; \
        } \
    } while (0)

/**
 * Main function: Loads an image, upscales it using CUDA, and saves the result.
 */
int main() {
    // Load the input image
    cv::Mat input = cv::imread("input_tile.jpg");
    if (input.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Calculate dimensions
    int in_w = input.cols;
    int in_h = input.rows;
    int scale = 2;  // Upscaling factor
    int out_w = in_w * scale;
    int out_h = in_h * scale;

    // Create output image
    cv::Mat output(out_h, out_w, CV_8UC3);

    // Allocate device memory
    uchar *d_input, *d_output;
    size_t input_size = in_w * in_h * 3 * sizeof(uchar);
    size_t output_size = out_w * out_h * 3 * sizeof(uchar);

    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data, input_size, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((in_w + blockDim.x - 1) / blockDim.x, (in_h + blockDim.y - 1) / blockDim.y);
    bicubicUpscaleKernel<<<gridDim, blockDim>>>(d_input, d_output, in_w, in_h, scale);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output.data, d_output, output_size, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    // Save output image
    cv::imwrite("output_tile.jpg", output);

    return 0;
}
