#include "cuda_shim.h"
#include <gtest/gtest.h>

class MetalShimTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize the shim before each test
    cudaError_t init_result = cudaShimInit();
    if (init_result != cudaSuccess) {
      GTEST_SKIP() << "Metal not available, skipping tests";
    }
  }

  void TearDown() override {
    // Clean up after each test
    cudaShimShutdown();
  }
};

TEST_F(MetalShimTest, TestMemoryAllocation) {
  void *devPtr = nullptr;
  size_t size = 1024; // 1KB

  // Test memory allocation
  EXPECT_EQ(cudaMalloc(&devPtr, size), cudaSuccess);
  EXPECT_NE(devPtr, nullptr);

  // Test memory free
  if (devPtr) {
    EXPECT_EQ(cudaFree(devPtr), cudaSuccess);
  }
}

TEST_F(MetalShimTest, TestMemcpy) {
  const size_t size = 1024;
  float *h_src = new float[size];
  float *h_dst = new float[size];
  float *d_ptr = nullptr;

  // Initialize host data
  for (size_t i = 0; i < size; ++i) {
    h_src[i] = static_cast<float>(i);
  }

  // Allocate device memory
  ASSERT_EQ(cudaMalloc((void **)&d_ptr, size * sizeof(float)), cudaSuccess);

  // Copy host to device
  EXPECT_EQ(
      cudaMemcpy(d_ptr, h_src, size * sizeof(float), cudaMemcpyHostToDevice),
      cudaSuccess);

  // Copy device to host
  EXPECT_EQ(
      cudaMemcpy(h_dst, d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost),
      cudaSuccess);

  // Verify data
  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(h_src[i], h_dst[i]);
  }

  // Clean up
  delete[] h_src;
  delete[] h_dst;
  if (d_ptr) {
    cudaFree(d_ptr);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
