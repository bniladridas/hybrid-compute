/**
 * Bicubic Image Upscaling Tool
 *
 * This program performs 2x bicubic upscaling on images using CUDA for GPU acceleration
 * or falls back to CPU with OpenMP parallelization. Bicubic interpolation provides
 * high-quality upscaling by considering a 4x4 neighborhood of pixels around each
 * target location and applying cubic interpolation in both x and y directions.
 *
 * Usage: ./upscaler [input_file] [output_file] [scale]
 */

/**
 * Bicubic Image Upscaling Tool
 *
 * This program performs 2x bicubic upscaling on images using CUDA for GPU acceleration
 * or falls back to CPU with OpenMP parallelization. Bicubic interpolation provides
 * high-quality upscaling by considering a 4x4 neighborhood of pixels around each
 * target location and applying cubic interpolation in both x and y directions.
 *
 * Usage: ./upscaler [input_file] [output_file] [scale]
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <exception>
#include <cstring>
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Clamp function for readability
__host__ __device__ float clamp(float value, float min_val, float max_val) {
    return min(max(value, min_val), max_val);
}

// Cubic interpolation function for 1D (Catmull-Rom spline)
__host__ __device__ float cubicInterpolate(float p0, float p1, float p2, float p3, float t) {
    return p1 + 0.5f * t * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + t * (3.0f * (p1 - p2) + p3 - p0));
}

// Perform bicubic interpolation on a pre-fetched 4x4 neighborhood
__host__ __device__ float perform_interpolation(const float vals[4][4], float tx, float ty) {
    float col[4];
    for (int m = 0; m < 4; m++) {
        col[m] = cubicInterpolate(vals[m][0], vals[m][1], vals[m][2], vals[m][3], tx);
    }
    float value = cubicInterpolate(col[0], col[1], col[2], col[3], ty);
    return fminf(fmaxf(value, 0.0f), 255.0f);
}
    float value = cubicInterpolate(col[0], col[1], col[2], col[3], ty);
    return clamp(value, 0.0f, 255.0f);
}

// Fetch the 4x4 neighborhood values for a given channel (CPU version)
__host__ __device__ void fetchVals(uchar* input, int in_w, int in_h, int channels, int gxi, int gyi, int c, float vals[4][4]) {
    for (int m = -1; m <= 2; m++) {
        for (int n = -1; n <= 2; n++) {
            int px = clamp(gxi + m, 0, in_w - 1);
            int py = clamp(gyi + n, 0, in_h - 1);
            vals[m + 1][n + 1] = input[(py * in_w + px) * channels + c];
        }
    }
}

// Fetch using texture memory (GPU version)
__device__ void fetchValsTex(cudaTextureObject_t* texObjs, int in_w, int in_h, int gxi, int gyi, int c, float vals[4][4]) {
    for (int m = -1; m <= 2; m++) {
        for (int n = -1; n <= 2; n++) {
            int px = clamp(gxi + m, 0, in_w - 1);
            int py = clamp(gyi + n, 0, in_h - 1);
            vals[m + 1][n + 1] = tex2D<unsigned char>(texObjs[c], px, py);
        }
    }
}
    }
}

// Compute bicubic interpolated value at (gx, gy) for channel c
__host__ __device__ float getBicubicValue(uchar* input, int in_w, int in_h, int channels, float gx, float gy, int c) {
    int gxi = (int)gx;
    int gyi = (int)gy;

    float vals[4][4];
    fetchVals(input, in_w, in_h, channels, gxi, gyi, c, vals);

    float tx = gx - gxi;
    float ty = gy - gyi;
    return perform_interpolation(vals, tx, ty);
}

// CUDA kernel for bicubic upscaling using texture memory
__global__ void bicubicUpscaleKernel(cudaTextureObject_t* texObjs, uchar* output, int in_w, int in_h, int channels, int scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in_w || y >= in_h) return;

    int gxi = x;
    int gyi = y;

    for (int c = 0; c < channels; c++) {
        float vals[4][4];
        fetchValsTex(texObjs, in_w, in_h, gxi, gyi, c, vals);

        for (int i = 0; i < scale; i++) {
            for (int j = 0; j < scale; j++) {
                float gx = (float)x + (float)i / (float)scale;
                float gy = (float)y + (float)j / (float)scale;
                float tx = gx - gxi;
                float ty = gy - gyi;

                float value = perform_interpolation(vals, tx, ty);

                int out_idx = ((y * scale + j) * (in_w * scale) + (x * scale + i)) * channels + c;
                output[out_idx] = (uchar)value;
            }
        }
    }
}

// Macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            return -1; \
        } \
    } while (0)

// Main function: performs bicubic upscaling using CUDA if available, else CPU
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

    // Get image properties
    int in_w = input.cols;
    int in_h = input.rows;
    int channels = input.channels();
    int out_w = in_w * scale;
    int out_h = in_h * scale;

    // Validate inputs
    if (channels < 1 || channels > 4) {
        std::cerr << "Error: Unsupported number of channels: " << channels << std::endl;
        return -1;
    }

    // Create output image
    cv::Mat output(out_h, out_w, input.type());

    // Check for GPU availability
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    bool useGPU = deviceCount > 0;

    // Allocate memory: use unified memory for output, textures for input on GPU
    uchar *d_output;
    size_t output_size = out_w * out_h * channels * sizeof(uchar);
    cudaTextureObject_t texObjs[4]; // max 4 channels
    cudaArray* cuArrays[4];

    if (useGPU) {
        // Split input into channels and create textures
        std::vector<cv::Mat> channelMats;
        cv::split(input, channelMats);

        for (int c = 0; c < channels; c++) {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
            CUDA_CHECK(cudaMallocArray(&cuArrays[c], &channelDesc, in_w, in_h));
            CUDA_CHECK(cudaMemcpy2DToArray(cuArrays[c], 0, 0, channelMats[c].data, in_w, in_w, in_h, cudaMemcpyHostToDevice));

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuArrays[c];

            cudaTextureDesc texDesc = {};
            texDesc.addressMode[0] = cudaAddressModeClamp;
            texDesc.addressMode[1] = cudaAddressModeClamp;
            texDesc.filterMode = cudaFilterModePoint;
            texDesc.readMode = cudaReadModeElementType;

            CUDA_CHECK(cudaCreateTextureObject(&texObjs[c], &resDesc, &texDesc, nullptr));
        }

        CUDA_CHECK(cudaMallocManaged(&d_output, output_size));
    } else {
        d_output = output.data;
    }

    if (useGPU) {
        // Launch CUDA kernel with 16x16 thread blocks
        dim3 blockDim(16, 16);
        dim3 gridDim((in_w + blockDim.x - 1) / blockDim.x, (in_h + blockDim.y - 1) / blockDim.y);
        bicubicUpscaleKernel<<<gridDim, blockDim>>>(texObjs, d_output, in_w, in_h, channels, scale);

        // Synchronize and check for kernel errors
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back to host
        memcpy(output.data, d_output, output_size);
    } else {
        // CPU fallback with OpenMP parallelization
        #pragma omp parallel for
        for (int y_out = 0; y_out < out_h; y_out++) {
            for (int x_out = 0; x_out < out_w; x_out++) {
                float gx = (float)x_out / (float)scale;
                float gy = (float)y_out / (float)scale;
                for (int c = 0; c < channels; c++) {
                    float value = getBicubicValue((uchar*)input.data, in_w, in_h, channels, gx, gy, c);
                    int out_idx = (y_out * out_w + x_out) * channels + c;
                    d_output[out_idx] = (uchar)value;
                }
            }
        }
    }

    // Free memory if GPU
    if (useGPU) {
        for (int c = 0; c < channels; c++) {
            CUDA_CHECK(cudaDestroyTextureObject(texObjs[c]));
            CUDA_CHECK(cudaFreeArray(cuArrays[c]));
        }
        CUDA_CHECK(cudaFree(d_output));
    }

    // Save output image
    cv::imwrite(output_file, output);

    std::cout << "Upscaling complete. Output saved to " << output_file << std::endl;

    return 0;
}
