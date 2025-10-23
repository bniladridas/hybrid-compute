/**
 * CUDA Canny Edge Detection
 */

// Placeholder
__global__ void cannyKernel(uchar* input, uchar* output, int width, int height) {
    // TODO
}

int applyCanny(const cv::Mat& input, cv::Mat& output) {
    output = input.clone();
    return 0;
}

int main(int argc, char** argv) {
    cv::Mat input = cv::imread(argv[1]);
    cv::Mat output;
    applyCanny(input, output);
    cv::imwrite(argv[2], output);
    std::cout << "Canny applied." << std::endl;
    return 0;
}
