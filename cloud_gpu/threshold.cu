/**
 * CUDA Thresholding
 */

// Placeholder
__global__ void thresholdKernel(uchar *input, uchar *output, int width,
                                int height, int thresh) {
  // TODO
}

int applyThresholding(const cv::Mat &input, cv::Mat &output, int thresh) {
  output = input.clone();
  return 0;
}

int main(int argc, char **argv) {
  cv::Mat input = cv::imread(argv[1]);
  cv::Mat output;
  int thresh = atoi(argv[3]);
  applyThresholding(input, output, thresh);
  cv::imwrite(argv[2], output);
  std::cout << "Thresholding applied." << std::endl;
  return 0;
}
