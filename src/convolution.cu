//------------------------------------------------------------------------------
// 2D Convolution Implementation comparing CPU and GPU performance
//------------------------------------------------------------------------------
// This file implements both CPU and GPU versions of 2D convolution with a
// configurable kernel size. It includes performance measurement and result validation.
//
// Compilation:
// nvcc -O3 -arch=sm_86 -std=c++20 --use_fast_math -Xcompiler "-fopenmp -fPIC -pthread -march=native" -I. convolution.cu -o convolution
//
// Usage:
//   ./convolution <width> <height> <kernel_radius>
// Example:
//   ./convolution 1024 1024 3  # 1024x1024 image with 7x7 kernel
//------------------------------------------------------------------------------

// Include custom GPU utilities for timing and validation
#include "gpu_utils.cuh"
// Include OpenMP for parallel CPU implementation and timing
#include <omp.h>
// Add these includes at the top
#include <vector>
#include <string>

//------------------------------------------------------------------------------
// Test Configuration
//------------------------------------------------------------------------------
// Type alias for test registry with convolution parameter types
using ConvTestRegistry = TestRegistry<const float*, const float*, float*, 
                                    int, int, int>;

// Create test registry with CPU testing enabled
static ConvTestRegistry conv_tests("2D Convolution", 1e-5f, false);

// Forward declare all test implementations
PerfMetrics runCPUTest(const float*, const float*, float*, int, int, int);
PerfMetrics runConvolutionTest(const float*, const float*, float*, int, int, int);
PerfMetrics runSharedMemoryTest(const float*, const float*, float*, int, int, int);
PerfMetrics runRegisterTiledTest(const float*, const float*, float*, int, int, int);

// Initialize all tests
void initializeTests() {
    conv_tests.addTest("CPU Reference", runCPUTest, false, true);  // CPU test, disabled
    conv_tests.addTest("Naive GPU", runConvolutionTest, true);     // Test 0
    conv_tests.addTest("Shared Memory", runSharedMemoryTest, true); // Test 1
    conv_tests.addTest("Register Tiled", runRegisterTiledTest, true); // Test 2
}

//------------------------------------------------------------------------------
// Basic GPU Convolution Kernel
// Parameters:
//   input: Input image data
//   kernel: Convolution kernel coefficients
//   output: Output image data
//   r: Kernel radius (kernel size = 2r + 1)
//   width, height: Image dimensions
//------------------------------------------------------------------------------
__global__ void conv2d_basic_kernel(const float *input, const float *kernel, float *output, 
                                  int r, int width, int height) {
    // Calculate output pixel coordinates
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize accumulator for convolution sum
    float Pvalue = 0;
    int inRow, inCol;

    // Iterate over kernel dimensions
    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol = 0; fCol < 2*r+1; fCol++) {
            // Calculate input image coordinates
            inRow = outRow - r + fRow;
            inCol = outCol - r + fCol;

            // Check boundaries and accumulate convolution
            if (inRow>=0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += kernel[fRow*(2*r+1) + fCol] * input[inRow*width + inCol];
            }
        }
    }

    // Write output if within image boundaries
    if (outRow < height && outCol < width) {
        output[outRow*width + outCol] = Pvalue;
    }
}

//------------------------------------------------------------------------------
// GPU Test Runner
// Configures and launches the GPU convolution kernel
// Parameters:
//   h_input: Host input image
//   h_kernel: Host convolution kernel
//   h_output: Host output buffer
//   width, height: Image dimensions
//   kernel_radius: Convolution kernel radius
// Returns:
//   PerfMetrics: Structure containing timing and performance data
//------------------------------------------------------------------------------
PerfMetrics runConvolutionTest(const float* h_input, const float* h_kernel, float* h_output,
                              int width, int height, int kernel_radius) {
    // Configure kernel launch parameters
    dim3 block(32, 32);
    dim3 grid(ceil_div(width, block.x), ceil_div(height, block.y));
    LaunchConfig config(grid, block);  // No shared memory
    
    // Calculate memory sizes
    size_t input_size = width * height;
    size_t kernel_size = (2*kernel_radius + 1) * (2*kernel_radius + 1);
    size_t output_size = width * height;
    
    // Calculate FLOPS (2 operations per multiply-add)
    size_t flops_per_thread = 2 * kernel_size;
    
    // Launch GPU test
    return runGpuTest<float>(
        conv_tests.getCurrentTestName(),  // Will now be "Test 0: Naive GPU"
        conv2d_basic_kernel,
        h_input, h_kernel, h_output,
        input_size, kernel_size, output_size,
        config,
        flops_per_thread,
        kernel_radius, width, height
    );
}

//------------------------------------------------------------------------------
// CPU Reference Implementation
// Provides baseline results for validation
// Parameters:
//   input: Input image
//   kernel: Convolution kernel
//   output: Output image
//   width, height: Image dimensions
//   kernel_radius: Convolution kernel radius
//------------------------------------------------------------------------------
void conv2d_cpu_reference(const float* input, const float* kernel, float* output,
                         int width, int height, int kernel_radius) {
    // Iterate over each output pixel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            // Apply convolution kernel
            for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
                for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
                    int in_y = y + ky;
                    int in_x = x + kx;
                    // Check boundaries
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

// Fix the shared memory kernel declaration
__global__ void conv2d_shared_kernel(const float* input, const float* kernel, float* output,
                                   int r, int width, int height) {
    // Shared memory for input tile
    // Add padding of radius size on each side
    extern __shared__ float shared_input[];
    
    // Calculate thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int x = bx + tx;
    int y = by + ty;
    
    // Calculate dimensions of the shared memory tile
    const int BLOCK_WIDTH = blockDim.x;
    const int BLOCK_HEIGHT = blockDim.y;
    const int TILE_WIDTH = BLOCK_WIDTH + 2 * r;
    const int TILE_HEIGHT = BLOCK_HEIGHT + 2 * r;
    
    // Load input tile into shared memory including halo region
    for (int row = ty; row < TILE_HEIGHT; row += BLOCK_HEIGHT) {
        for (int col = tx; col < TILE_WIDTH; col += BLOCK_WIDTH) {
            int global_row = by + row - r;
            int global_col = bx + col - r;
            
            // Check boundaries and load data
            if (global_row >= 0 && global_row < height && 
                global_col >= 0 && global_col < width) {
                shared_input[row * TILE_WIDTH + col] = 
                    input[global_row * width + global_col];
            } else {
                shared_input[row * TILE_WIDTH + col] = 0.0f;
            }
        }
    }
    
    // Ensure all threads have loaded their data
    __syncthreads();
    
    // Compute convolution only for valid output pixels
    if (x < width && y < height) {
        float sum = 0.0f;
        
        // Perform convolution using shared memory
        for (int ky = 0; ky < 2*r + 1; ky++) {
            for (int kx = 0; kx < 2*r + 1; kx++) {
                // Calculate position in shared memory
                int shared_row = ty + ky;
                int shared_col = tx + kx;
                
                sum += kernel[ky * (2*r + 1) + kx] * 
                       shared_input[shared_row * TILE_WIDTH + shared_col];
            }
        }
        
        // Write output
        output[y * width + x] = sum;
    }
}

PerfMetrics runSharedMemoryTest(const float* h_input, const float* h_kernel, float* h_output,
                               int width, int height, int kernel_radius) {
    // Configure kernel launch parameters
    dim3 block(32, 32);
    dim3 grid(ceil_div(width, block.x), ceil_div(height, block.y));
    
    // Calculate shared memory size
    const int TILE_WIDTH = block.x + 2 * kernel_radius;
    const int TILE_HEIGHT = block.y + 2 * kernel_radius;
    size_t shared_mem_size = TILE_WIDTH * TILE_HEIGHT * sizeof(float);
    
    // Create launch config with shared memory
    LaunchConfig config(grid, block, shared_mem_size);
    
    // Calculate memory sizes
    size_t input_size = width * height;
    size_t kernel_size = (2*kernel_radius + 1) * (2*kernel_radius + 1);
    size_t output_size = width * height;
    
    // Calculate FLOPS (2 operations per multiply-add)
    size_t flops_per_thread = 2 * kernel_size;
    
    // Launch GPU test
    return runGpuTest<float>(
        conv_tests.getCurrentTestName(),  // Will now be "Test 1: Shared Memory"
        conv2d_shared_kernel,
        h_input, h_kernel, h_output,
        input_size, kernel_size, output_size,
        config,
        flops_per_thread,
        kernel_radius, width, height
    );
}

// Wrap CPU implementation in same interface as GPU tests
PerfMetrics runCPUTest(const float* h_input, const float* h_kernel, float* h_output,
                       int width, int height, int kernel_radius) {
    PerfMetrics pm = {0};
    
    double start = omp_get_wtime();
    conv2d_cpu_reference(h_input, h_kernel, h_output, width, height, kernel_radius);
    double end = omp_get_wtime();
    
    pm.totalTime = (end - start) * 1000.0;  // Convert to ms
    pm.gflops = (2.0 * width * height * (2*kernel_radius+1) * (2*kernel_radius+1)) / 
                (pm.totalTime * 1e6);
    
    return pm;
}

// Fix the register tiling kernel implementation
__global__ void conv2d_register_tiled_kernel(const float *input, const float *kernel, 
                                           float *output, int r, int width, int height) {
    // Calculate base coordinates (each thread handles 2x2 pixels)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * (blockDim.x * 2); // Multiply by 2 for tile width
    const int by = blockIdx.y * (blockDim.y * 2); // Multiply by 2 for tile height
    const int x = bx + tx * 2; // Multiply by 2 to space out threads
    const int y = by + ty * 2;

    // Constants for register tiling
    constexpr int TILE_X = 2;
    constexpr int TILE_Y = 2;
    
    // Compute convolution for each pixel in the tile
    #pragma unroll
    for (int ty = 0; ty < TILE_Y; ty++) {
        #pragma unroll
        for (int tx = 0; tx < TILE_X; tx++) {
            const int out_y = y + ty;
            const int out_x = x + tx;
            
            // Only compute if within bounds
            if (out_y < height && out_x < width) {
                float sum = 0.0f;
                
                // Apply convolution kernel
                for (int ky = -r; ky <= r; ky++) {
                    for (int kx = -r; kx <= r; kx++) {
                        const int in_y = out_y + ky;
                        const int in_x = out_x + kx;
                        
                        if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                            sum += input[in_y * width + in_x] * 
                                  kernel[(ky + r) * (2*r + 1) + (kx + r)];
                        }
                    }
                }
                
                output[out_y * width + out_x] = sum;
            }
        }
    }
}

// Fix the test runner for register tiling implementation
PerfMetrics runRegisterTiledTest(const float* h_input, const float* h_kernel, 
                                float* h_output, int width, int height, 
                                int kernel_radius) {
    // Configure grid and block sizes
    dim3 block(16, 16);  // Each thread handles 2x2 pixels
    dim3 grid(ceil_div(width, block.x * 2), ceil_div(height, block.y * 2));
    LaunchConfig config(grid, block);
    
    size_t input_size = width * height;
    size_t kernel_size = (2*kernel_radius + 1) * (2*kernel_radius + 1);
    size_t output_size = width * height;
    size_t flops_per_thread = 2 * kernel_size * 4; // 4 pixels per thread
    
    return runGpuTest<float>(
        conv_tests.getCurrentTestName(),
        conv2d_register_tiled_kernel,
        h_input, h_kernel, h_output,
        input_size, kernel_size, output_size,
        config,
        flops_per_thread,
        kernel_radius, width, height
    );
}

//------------------------------------------------------------------------------
// Main Function
// Orchestrates the convolution test:
// 1. Parses command line arguments
// 2. Allocates and initializes memory
// 3. Runs CPU reference implementation
// 4. Runs GPU implementation
// 5. Validates results
// 6. Reports performance metrics
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Validate command line arguments
    if (argc != 4) {
        printf("Usage: ./convolution <width> <height> <kernel_radius>\n");
        return 1;
    }

    // Parse and validate input parameters
    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int kernel_radius = atoi(argv[3]);

    if (width <= 0 || height <= 0 || kernel_radius <= 0) {
        printf("Error: All dimensions must be positive integers.\n");
        return 1;
    }

    // Calculate memory requirements
    size_t input_size = width * height;
    size_t kernel_size = (2 * kernel_radius + 1) * (2 * kernel_radius + 1);

    // Allocate host memory
    float *h_input = (float*)malloc(input_size * sizeof(float));
    float *h_kernel = (float*)malloc(kernel_size * sizeof(float));
    float *h_output = (float*)malloc(input_size * sizeof(float));
    float *h_output_cpu = (float*)malloc(input_size * sizeof(float));

    // Verify memory allocation
    if (!h_input || !h_kernel || !h_output || !h_output_cpu) {
        printf("Error: Host memory allocation failed.\n");
        return 1;
    }

    // Initialize input data with random values
    srand(time(NULL));
    for (size_t i = 0; i < input_size; i++) {
        h_input[i] = (float)(rand()) / RAND_MAX;
    }

    // Initialize convolution kernel with random values
    for (size_t i = 0; i < kernel_size; i++) {
        h_kernel[i] = (float)(rand()) / RAND_MAX;
    }

    initializeTests();
    
    char dimensions[256];
    snprintf(dimensions, sizeof(dimensions), 
             "Input size: %d x %d, Kernel: %d x %d",
             width, height, 2*kernel_radius+1, 2*kernel_radius+1);

    // Run all tests
    conv_tests.runAll(dimensions, h_input, h_kernel, h_output, width, height, kernel_radius);

    // Free allocated memory
    free(h_input);
    free(h_kernel);
    free(h_output);
    free(h_output_cpu);

    return 0;
}

float checkResults(const float* baseline, const float* test, int total_elements, 
                  float tol, const char* impl_name = nullptr) {
    float max_diff = 0.0f;
    for (int i = 0; i < total_elements; i++) {
        float diff = fabs(baseline[i] - test[i]);
        max_diff = max(max_diff, diff);
    }
    
    if (impl_name) {
        printf("%s: Accuracy (max diff: %e)\n", impl_name, max_diff);
    }
    printf("   Accuracy Check: %s (max diff: %e)\n", 
           max_diff <= tol ? "PASSED" : "FAILED", max_diff);
    return max_diff;
}




    