/**
 * CUDA Image Sharpening
 */

// Placeholder
__global__ void sharpenKernel(uchar* input, uchar* output, int width, int height) {
    // TODO
}

int applySharpening(const cv::Mat& input, cv::Mat& output) {
    output = input.clone();
    return 0;
}

int main(int argc, char** argv) {
    cv::Mat input = cv::imread(argv[1]);
    cv::Mat output;
    applySharpening(input, output);
    cv::imwrite(argv[2], output);
    std::cout << "Sharpening applied." << std::endl;
    return 0;
}
