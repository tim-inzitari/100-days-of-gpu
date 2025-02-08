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

// First, create a CPU reference implementation for baseline results
void conv2d_cpu_reference(const float* input, const float* kernel, float* output,
                         int width, int height, int kernel_radius) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
                for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
                    int in_y = y + ky;
                    int in_x = x + kx;
                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                        sum += input[in_y * width + in_x] * 
                               kernel[(ky + kernel_radius) * (2*kernel_radius + 1) + 
                                    (kx + kernel_radius)];
                    }
                }
            }
            output[y * width + x] = sum;
        }
    }
}

// Then in your main function or test harness:
int main() {
    // ... setup code ...
    
    // Allocate memory for results
    float *h_output = (float*)malloc(width * height * sizeof(float));
    float *h_output_cpu = (float*)malloc(width * height * sizeof(float));
    
    // Generate reference result
    conv2d_cpu_reference(h_input, h_kernel, h_output_cpu, 
                        width, height, kernel_radius);
    
    // Run GPU implementation
    PerfMetrics pm = runConvolutionTest(h_input, h_kernel, h_output,
                                      width, height, kernel_radius);
    
    // Validate results
    const float tolerance = 1e-5f;  // Adjust based on precision requirements
    checkResults(h_output_cpu,      // CPU reference result
                h_output,           // GPU result to validate
                width * height,     // Total number of elements
                tolerance,          // Maximum allowed difference
                "2D Convolution"    // Implementation name
    );
    
    // ... cleanup code ...
    free(h_output);
    free(h_output_cpu);
}




    