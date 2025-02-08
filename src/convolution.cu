#include "gpu_utils.cuh"

__global__ void conv2d_basic_kernel(const float *input, const float *kernel, float *output, 
                                  int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0;
    int inRow, inCol;

    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol = 0; fCol < 2*r+1; fCol++) {
            inRow = outRow - r + fRow;
            inCol = outCol - r + fCol;

            if (inRow>=0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += kernel[fRow*(2*r+1) + fCol] * input[inRow*width + inCol];
            }
        }
    }

    if (outRow < height && outCol < width) {
        output[outRow*width + outCol] = Pvalue;
    }
}

PerfMetrics runConvolutionTest(const float* h_input, const float* h_kernel, float* h_output,
                              int width, int height, int kernel_radius) {
    dim3 block(16, 16);
    dim3 grid((width + 15)/16, (height + 15)/16);
    
    size_t input_size = width * height;
    size_t kernel_size = (2*kernel_radius + 1) * (2*kernel_radius + 1);
    size_t output_size = width * height;
    
    // Each thread does (2*r+1)*(2*r+1) multiply-adds
    size_t flops_per_thread = 2 * kernel_size;
    
    return runGpuTest<float>(
        "Convolution Test",
        conv2d_basic_kernel,
        h_input, h_kernel, h_output,
        input_size, kernel_size, output_size,
        grid, block,
        flops_per_thread,
        kernel_radius, width, height  // Additional kernel parameters
    );
}




    