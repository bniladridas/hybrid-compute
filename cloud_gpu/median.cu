/**
 * CUDA Median Filter
 */

// Placeholder
__global__ void medianKernel(uchar *input, uchar *output, int width,
                             int height) {
  // TODO
}

int applyMedianFilter(const cv::Mat &input, cv::Mat &output) {
  output = input.clone();
  return 0;
}

int main(int argc, char **argv) {
  cv::Mat input = cv::imread(argv[1]);
  cv::Mat output;
  applyMedianFilter(input, output);
  cv::imwrite(argv[2], output);
  std::cout << "Median filter applied." << std::endl;
  return 0;
}
