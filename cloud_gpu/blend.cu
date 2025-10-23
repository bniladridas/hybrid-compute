/**
 * CUDA Image Blending
 */

// Placeholder
__global__ void blendKernel(uchar* input1, uchar* input2, uchar* output, int width, int height, float alpha) {
    // TODO
}

int applyBlending(const cv::Mat& input1, const cv::Mat& input2, cv::Mat& output, float alpha) {
    output = input1.clone();
    return 0;
}

int main(int argc, char** argv) {
    cv::Mat input1 = cv::imread(argv[1]);
    cv::Mat input2 = cv::imread(argv[2]);
    cv::Mat output;
    float alpha = atof(argv[4]);
    applyBlending(input1, input2, output, alpha);
    cv::imwrite(argv[3], output);
    std::cout << "Blending applied." << std::endl;
    return 0;
}
