#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <tuple>  // Add this for tuple support

//------------------------------------------------------------------------------
// Contents
//------------------------------------------------------------------------------
// Structures:
//   PerfMetrics     - Performance measurement data for GPU operations
//   TestResult      - Test result data including name and metrics
//   KernelTest      - Test definition for registry system
//   LaunchConfig    - CUDA kernel launch configuration (grid, block, shared memory)
//
// Classes:
//   TestRegistry    - Test management and execution system
//
// Enums:
//   CompareMode     - Performance comparison modes (BASE_ONLY, VS_CPU)
//
// Functions:
//   runGpuTest      - Run and measure GPU kernel performance
//   checkResults    - Compare and validate implementation results
//   printPerformanceSummary - Format and display test results
//
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Global Configuration Flags
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GPU Performance Metrics Structure Documentation Block
//------------------------------------------------------------------------------
// Usage example:
//   PerfMetrics metrics = runGpuTest<float>(
//       "Test Name",
//       kernel_func,
//       h_A, h_B, h_C,
//       size_A, size_B, size_C,
//       config,  // LaunchConfig with grid, block, and shared memory
//       flops_per_thread,
//       additional_args...);  // Update example to use LaunchConfig
//   printf("Kernel time: %.3f ms\n", metrics.kernelTime);
//   if (metrics.gflops < expected_gflops) {
//       printf("Performance below expected threshold\n");
//   }
//------------------------------------------------------------------------------
struct PerfMetrics
{
    // Time taken to transfer data from CPU to GPU (H2D)
    float transferTime;
    // Time taken by the GPU kernel to execute
    float kernelTime;
    // Time taken to transfer results back to CPU (D2H)
    float d2hTime;
    // Sum of transfer and kernel times
    float totalTime;
    // Computational throughput in billion floating point operations per second
    float gflops;
};

//------------------------------------------------------------------------------
// Performance Result Structures Documentation Block
//------------------------------------------------------------------------------
// Usage example:
//   TestResult result = {
//       "GPU Implementation", 
//       {1.2f, 0.5f, 0.3f, 2.0f, 150.0f},  // PerfMetrics
//       true  // valid
//   };
//   if (result.valid) {
//       printf("Implementation %s achieved %.2f GFLOPS\n", 
//              result.name, result.metrics.gflops);
//   }
//------------------------------------------------------------------------------
struct TestResult {
    const char* name;      // Implementation name for reporting
    PerfMetrics metrics;   // Performance measurements
    bool valid;            // Whether this result should be included in summary
};

//------------------------------------------------------------------------------
// Performance Summary Modes
//------------------------------------------------------------------------------
// Defines different modes for comparing and reporting performance results.
// BASE_ONLY compares all implementations against a baseline implementation.
// VS_CPU additionally includes comparisons against a CPU reference implementation.
//
// Usage example:
//   printPerformanceSummary<CompareMode::VS_CPU>(
//       "Matrix Multiplication", "1024x1024",
//       results, num_results, baseline, cpu_result);
//------------------------------------------------------------------------------
enum class CompareMode {
    BASE_ONLY,  // Compare implementations against baseline only
    VS_CPU      // Show comparisons against both baseline and CPU
};

//------------------------------------------------------------------------------
// Launch Configuration Structure Documentation Block
//------------------------------------------------------------------------------
// The LaunchConfig struct provides a unified way to specify CUDA kernel launch 
// parameters including grid dimensions, block dimensions, and shared memory size.
//
// Members:
//   grid: CUDA grid dimensions (dim3)
//   block: CUDA block dimensions (dim3)
//   shared_mem: Shared memory size in bytes (default: 0)
//
// Usage Example 1 - Basic Configuration:
//   dim3 block(32, 32);
//   dim3 grid((width + 31)/32, (height + 31)/32);
//   LaunchConfig config(grid, block);
//
// Usage Example 2 - With Shared Memory:
//   dim3 block(16, 16);
//   dim3 grid((width + 15)/16, (height + 15)/16);
//   size_t shared_mem_size = (block.x + 2*radius) * (block.y + 2*radius) * sizeof(float);
//   LaunchConfig config(grid, block, shared_mem_size);
//
// Notes:
//   - Grid and block dimensions should be calculated to cover the entire output
//   - Shared memory size is optional and defaults to 0
//   - Block size should consider occupancy and hardware limitations
//------------------------------------------------------------------------------

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem = 0;  // Default to 0 for no shared memory
    
    LaunchConfig(dim3 g, dim3 b, size_t sm = 0) 
        : grid(g), block(b), shared_mem(sm) {}
};

//------------------------------------------------------------------------------
// Main GPU Test Runner Documentation Block
//------------------------------------------------------------------------------
// This template function provides a standardized way to:
// 1. Measure GPU kernel performance
// 2. Handle memory transfers
// 3. Calculate GFLOPS
// 4. Clean up resources
//
// Template Parameters:
//   T: Data type for computation (float, double, half, etc.)
//   Args: Variable argument types for additional kernel parameters
//
// Function Parameters:
//   base_name: Name of the test
//   kernel: Pointer to the GPU kernel function
//   h_A: First input array on host
//   h_B: Second input array on host
//   h_C: Output array on host
//   size_A: Number of elements in first input array
//   size_B: Number of elements in second input array
//   size_C: Number of elements in output array
//   config: LaunchConfig containing grid, block, and shared memory configuration
//   flops_per_thread: Number of floating point operations per thread
//   args: Additional kernel arguments
//
// Returns:
//   PerfMetrics structure containing timing and performance data
//
// Usage Example 1 - Matrix Multiplication:
//   dim3 block(32, 32);
//   dim3 grid((M + 31)/32, (N + 31)/32);
//   LaunchConfig config(grid, block);
//   
//   size_t size_A = M * K;
//   size_t size_B = K * N;
//   size_t size_C = M * N;
//   size_t flops = 2 * K; // multiply-add per element
//
//   PerfMetrics results = runGpuTest<float>(
//       "Matrix Multiplication",
//       matmul_kernel,
//       h_A, h_B, h_C,
//       size_A, size_B, size_C,
//       config,
//       flops,
//       M, N, K  // Additional kernel parameters
//   );
//
// Usage Example 2 - Convolution with Shared Memory:
//   dim3 block(16, 16);
//   dim3 grid((width + 15)/16, (height + 15)/16);
//   size_t shared_mem_size = (block.x + 2*radius) * (block.y + 2*radius) * sizeof(float);
//   LaunchConfig config(grid, block, shared_mem_size);
//   
//   size_t size_input = width * height;
//   size_t size_kernel = kernel_width * kernel_height;
//   size_t size_output = width * height;
//   size_t flops = 2 * kernel_width * kernel_height; // multiply-add per element
//
//   PerfMetrics results = runGpuTest<float>(
//       "2D Convolution",
//       conv2d_kernel,
//       h_input, h_kernel, h_output,
//       size_input, size_kernel, size_output,
//       config,
//       flops,
//       width, height, kernel_radius
//   );
//
// Error Handling:
//   - The function includes basic CUDA error checking
//   - Memory allocation failures will be printed to stderr
//   - Kernel execution errors will be reported
//
// Performance Tips:
//   1. Choose appropriate block sizes (typically 128-512 threads)
//   2. Ensure grid dimensions cover entire output
//   3. Calculate flops_per_thread accurately for meaningful GFLOPS
//   4. Consider memory coalescing in kernel design
//
// Memory Management:
//   - Function handles all GPU memory allocation/deallocation
//   - Host memory should be allocated before calling
//   - All GPU resources are cleaned up before return
//------------------------------------------------------------------------------

template <typename T, typename... KernelArgs>
PerfMetrics runGpuTest(
    const char *base_name,
    void (*kernel)(const T *, const T *, T *, KernelArgs...),
    const T *h_A,
    const T *h_B,
    T *h_C,
    size_t size_A,
    size_t size_B,
    size_t size_C,
    const LaunchConfig& config,
    size_t flops_per_thread,
    KernelArgs... args)
{
    // Initialize performance metrics structure to zero
    PerfMetrics pm = {0};
    // Variable to store timing results
    float elapsed;
    // CUDA event objects for timing measurements
    cudaEvent_t start, stop;
    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Declare pointers for device memory
    T *d_A, *d_B, *d_C;
    // Allocate device memory for first input array
    cudaMalloc(&d_A, size_A * sizeof(T));
    // Allocate device memory for second input array
    cudaMalloc(&d_B, size_B * sizeof(T));
    // Allocate device memory for output array
    cudaMalloc(&d_C, size_C * sizeof(T));

    // Begin timing host to device transfers
    cudaEventRecord(start);
    // Copy first input array to device
    cudaMemcpy(d_A, h_A, size_A * sizeof(T), cudaMemcpyHostToDevice);
    // Copy second input array to device
    cudaMemcpy(d_B, h_B, size_B * sizeof(T), cudaMemcpyHostToDevice);
    // Mark end of transfer
    cudaEventRecord(stop);
    // Wait for transfer to complete
    cudaEventSynchronize(stop);
    // Calculate transfer time
    cudaEventElapsedTime(&elapsed, start, stop);
    // Store H2D transfer time
    pm.transferTime = elapsed;

    // Begin timing kernel execution
    cudaEventRecord(start);
    // Launch kernel with configured grid, block and shared memory
    kernel<<<config.grid, config.block, config.shared_mem>>>(d_A, d_B, d_C, args...);
    cudaEventRecord(stop);
    // Wait for kernel to complete
    cudaEventSynchronize(stop);
    // Calculate kernel execution time
    cudaEventElapsedTime(&elapsed, start, stop);
    // Store kernel execution time
    pm.kernelTime = elapsed;

    // Begin timing device to host transfer
    cudaEventRecord(start);
    // Copy results back to host
    cudaMemcpy(h_C, d_C, size_C * sizeof(T), cudaMemcpyDeviceToHost);
    // Mark end of transfer
    cudaEventRecord(stop);
    // Wait for transfer to complete
    cudaEventSynchronize(stop);
    // Calculate transfer time
    cudaEventElapsedTime(&elapsed, start, stop);
    // Store D2H transfer time
    pm.d2hTime = elapsed;

    // Calculate total execution time
    pm.totalTime = pm.transferTime + pm.kernelTime + pm.d2hTime;
    // Calculate total number of threads launched
    size_t total_threads = config.grid.x * config.grid.y * config.grid.z * config.block.x * config.block.y * config.block.z;
    // Calculate GFLOPS (billion floating point operations per second)
    pm.gflops = (flops_per_thread * total_threads) / (pm.kernelTime * 1e6f);

    // Print test name (already includes number)
    printf("%s:\n", base_name);
    printf("   H2D: %.3f ms, Kernel: %.3f ms, D2H: %.3f ms, Total: %.3f ms, GFLOPS: %.2f\n",
           pm.transferTime, pm.kernelTime, pm.d2hTime, pm.totalTime, pm.gflops);

    // Free device memory for first input array
    cudaFree(d_A);
    // Free device memory for second input array
    cudaFree(d_B);
    // Free device memory for output array
    cudaFree(d_C);
    // Destroy start timing event
    cudaEventDestroy(start);
    // Destroy stop timing event
    cudaEventDestroy(stop);

    // Return performance metrics
    return pm;
}

//------------------------------------------------------------------------------
// Utility function to compare results between implementations
//------------------------------------------------------------------------------
// This function compares the output of different implementations against a baseline
// to validate correctness and measure numerical accuracy. It's particularly useful
// for comparing GPU implementations against CPU reference implementations or
// comparing optimized versions against naive implementations.
//
// Parameters:
//   baseline: Reference result to compare against (usually from a verified implementation)
//   test: Result to validate from the implementation being tested
//   total_elements: Total number of elements to compare
//   tol: Maximum allowed difference between elements (tolerance)
//   impl_name: Optional name of the implementation being tested (for output)
//
// Returns:
//   float: Maximum absolute difference found between any pair of elements
//
// Usage Example 1 - Matrix Multiplication:
//   // After running CPU and GPU implementations
//   const float tolerance = 1e-5f;
//   size_t total_elements = M * N;
//   checkResults(
//       h_C_cpu,                              // CPU reference result
//       h_C_gpu,                             // GPU implementation result
//       total_elements,                      // M * N elements
//       tolerance,                           // Maximum allowed difference
//       "GPU Matrix Multiplication"          // Implementation name
//   );
//
// Usage Example 2 - 2D Convolution:
//   // After running reference and GPU implementations
//   const float tolerance = 1e-5f;
//   size_t total_elements = width * height;
//   float max_diff = checkResults(
//       h_output_cpu,                        // CPU reference result
//       h_output_gpu,                        // GPU implementation result
//       total_elements,                      // width * height elements
//       tolerance,                           // Maximum allowed difference
//       "2D Convolution GPU"                 // Implementation name
//   );
//   if (max_diff > tolerance) {
//       printf("Warning: Large numerical differences detected\n");
//   }
//
// Usage Example 3 - Comparing Different GPU Implementations:
//   // Compare optimized implementation against naive baseline
//   const float tolerance = 1e-5f;
//   checkResults(
//       h_output_baseline,                   // Naive GPU implementation
//       h_output_optimized,                  // Optimized GPU implementation
//       total_elements,                      // Number of elements
//       tolerance,                           // Maximum allowed difference
//       "Optimized GPU Implementation"       // Implementation name
//   );
//
// Notes:
//   - Uses absolute difference for comparison
//   - Prints both implementation-specific message and general accuracy check
//   - PASSED/FAILED status depends on provided tolerance
//   - Higher tolerance might be needed for lower precision implementations
//   - Common tolerance values:
//     * 1e-5f for single precision (float)
//     * 1e-12 for double precision
//     * 1e-2f for half precision or tensor cores
//   - Consider using higher tolerances when:
//     * Working with large matrices/tensors
//     * Using lower precision arithmetic
//     * Implementing algorithms with known numerical instability
//------------------------------------------------------------------------------
template <typename T>
float checkResults(const T* baseline,      // Reference implementation results
                  const T* test,           // Test implementation results
                  int total_elements,      // Number of elements to compare
                  float tol,              // Maximum allowed difference
                  const char* impl_name = nullptr)  // Optional implementation name
{
    // Track maximum difference found between baseline and test
    float max_diff = 0.0f;
    
    // Compare each element pair
    for (int i = 0; i < total_elements; i++) {
        // Calculate absolute difference for current element
        float diff = fabs(float(baseline[i] - test[i]));
        // Update max_diff if current difference is larger
        max_diff = max(max_diff, diff);
    }
    
    // If implementation name provided, print implementation-specific message
    if (impl_name) {
        printf("%s: Accuracy (max diff: %e)\n", impl_name, max_diff);
    }
    
    // Print general accuracy check result
    printf("   Accuracy Check: %s (max diff: %e)\n", 
           max_diff <= tol ? "PASSED" : "FAILED", max_diff);
    
    // Return maximum difference for caller to use if needed
    return max_diff;
}

//------------------------------------------------------------------------------
// Performance Summary Printer
//------------------------------------------------------------------------------
template<CompareMode Mode>
inline void printPerformanceSummary(
    const char* title,
    const char* dimensions,
    const TestResult* results,
    int num_results,
    const TestResult& baseline,
    const TestResult* cpu_result = nullptr)
{
    printf("\n=== %s Performance Summary ===\n", title);
    printf("%s\n", dimensions);
    printf("--------------------------------------------------------------------------------\n");
    
    if constexpr (Mode == CompareMode::BASE_ONLY) {
        printf("Implementation                Time (ms)        GFLOPS       vs Baseline\n");
        printf("--------------------------------------------------------------------------------\n");
        
        // Print baseline result
        printf("%-25s  %12.3f    %10.2f    %8.2fx\n",
               baseline.name,                    // Use baseline name
               baseline.metrics.totalTime,
               baseline.metrics.gflops,
               1.0f);
        
        // Print each test result with its own name
        for (int i = 0; i < num_results; i++) {
            if (results[i].valid) {
                printf("%-25s  %12.3f    %10.2f    %8.2fx\n",
                       results[i].name,          // Use result name from array
                       results[i].metrics.totalTime,
                       results[i].metrics.gflops,
                       baseline.metrics.totalTime / results[i].metrics.totalTime);
            }
        }
    }
    else if constexpr (Mode == CompareMode::VS_CPU) {
        if (!cpu_result) {
            printf("Error: CPU result required for VS_CPU mode\n");
            return;
        }
        
        printf("Implementation           Time (ms)        GFLOPS     vs Base    vs CPU\n");
        printf("--------------------------------------------------------------------------------\n");
        
        // Print CPU result first
        printf("%-20s  %12.3f    %10.2f    %8.2fx   %7.2fx\n",
               cpu_result->name,
               cpu_result->metrics.totalTime,
               cpu_result->metrics.gflops,
               baseline.metrics.totalTime / cpu_result->metrics.totalTime,
               1.0f);
               
        // Print baseline
        printf("%-20s  %12.3f    %10.2f    %8.2fx   %7.2fx\n",
               baseline.name,
               baseline.metrics.totalTime,
               baseline.metrics.gflops,
               1.0f,
               cpu_result->metrics.totalTime / baseline.metrics.totalTime);
        
        // Print other results
        for (int i = 0; i < num_results; i++) {
            if (results[i].valid) {
                printf("%-20s  %12.3f    %10.2f    %8.2fx   %7.2fx\n",
                       results[i].name,
                       results[i].metrics.totalTime,
                       results[i].metrics.gflops,
                       baseline.metrics.totalTime / results[i].metrics.totalTime,
                       cpu_result->metrics.totalTime / results[i].metrics.totalTime);
            }
        }
    }
    printf("\n");
}

//------------------------------------------------------------------------------
// Test Registry System Documentation Block
//------------------------------------------------------------------------------
// This system provides a flexible framework for registering, managing, and
// executing different implementations of the same algorithm. It handles test
// orchestration, result collection, and performance reporting automatically.
//
// Template Parameters:
//   Args: Variable argument types matching the test function signatures
//
// Features:
// - Automatic test registration and management
// - Consistent performance measurement
// - Optional CPU reference comparison
// - Flexible result reporting
// - Support for enabling/disabling tests
//
// Usage Example 1 - Basic GPU Tests:
//   // Define test registry
//   using MyTestRegistry = TestRegistry<float*, float*, int, int>;
//   static MyTestRegistry tests("Algorithm Name");
//
//   // Register implementations
//   tests.addTest("Naive GPU", runBasicTest);
//   tests.addTest("Optimized", runOptimizedTest);
//
//   // Run all tests
//   tests.runAll(dimensions, data_in, data_out, width, height);
//
// Usage Example 2 - With CPU Reference:
//   // Run CPU implementation
//   double cpu_time = measureCPUTime(...);
//   double gflops = calculateGFLOPS(...);
//   tests.setCPUResult("CPU Reference", cpu_time, gflops);
//
//   // Run GPU tests with CPU comparison
//   tests.runAll(dimensions, data_in, data_out, width, height);
//
// Usage Example 3 - Selective Testing:
//   // Register tests with some disabled
//   tests.addTest("Basic", runBasicTest);
//   tests.addTest("Experimental", runExperimentalTest, false);  // disabled
//   tests.addTest("Optimized", runOptimizedTest);
//
// Notes:
// - First registered test becomes the baseline for comparisons
// - CPU comparison is optional and controlled by SKIP_CPU_TEST
// - Tests can be selectively enabled/disabled during registration
// - Results include timing, GFLOPS, and relative performance metrics
//------------------------------------------------------------------------------

template<typename... Args>
class TestRegistry {
private:
    struct KernelTest {
        std::string name;              // Store full name with number
        PerfMetrics (*run)(Args...);
        bool enabled;
        bool isCPU;
    };

    std::vector<KernelTest> tests;
    std::string test_name;
    float tolerance;
    bool skip_cpu;
    TestResult cpu_result = {"", {0}, false};
    int current_test;  // Track current test being run
    float* baseline_output = nullptr;  // Store naive implementation results
    size_t output_size = 0;           // Size of output array

    // Helper to calculate output size for different operations
    size_t calculateOutputSize(const auto& args_tuple) {
        constexpr size_t tuple_size = std::tuple_size_v<std::remove_reference_t<decltype(args_tuple)>>;
        
        if (test_name == "2D Convolution") {
            // For convolution: width * height
            // Args are: const float*, const float*, float*, int width, int height, int kernel_radius
            return std::get<3>(args_tuple) * std::get<4>(args_tuple);  // width * height
        } else if (test_name == "Tensor Multiplication") {
            if constexpr (tuple_size >= 8) {
                // For tensor multiplication: batch_size * m * l
                // Args are: const float*, const float*, float*, int batch_size, int m, int n, int k, int l
                return std::get<3>(args_tuple) *  // batch_size
                       std::get<4>(args_tuple) *  // m
                       std::get<7>(args_tuple);   // l
            } else {
                // Handle case where tuple is too small
                return std::get<3>(args_tuple) * std::get<4>(args_tuple);
            }
        } else {
            // Default case (could be matrix multiplication or other)
            return std::get<3>(args_tuple) * std::get<4>(args_tuple);  // Default to M * N
        }
    }

public:
    TestRegistry(const char* name, float tol = 1e-5f, bool skip_cpu = false) 
        : test_name(name), tolerance(tol), skip_cpu(skip_cpu), current_test(0) {}

    void addTest(const char* name, PerfMetrics (*run)(Args...), 
                bool enabled = true, bool isCPU = false) {
        if (enabled && (!isCPU || !skip_cpu)) {
            char numbered_name[64];
            snprintf(numbered_name, sizeof(numbered_name), 
                    "Test %d: %s", (int)tests.size(), name);
            tests.push_back({numbered_name, run, true, isCPU});
        }
    }

    void runAll(const char* dimensions, Args... args) {
        current_test = 0;
        std::vector<TestResult> results;
        TestResult baseline;
        bool first = true;

        // Create tuple once at the beginning
        auto args_tuple = std::make_tuple(args...);

        for (const auto& test : tests) {
            PerfMetrics pm = test.run(args...);
            TestResult result = {test.name.c_str(), pm, true};
            current_test++;

            if (first && !test.isCPU) {  // Use first non-CPU test as baseline
                baseline = result;
                const float* output = std::get<2>(args_tuple);
                output_size = calculateOutputSize(args_tuple);
                
                if (baseline_output == nullptr) {
                    baseline_output = new float[output_size];
                }
                memcpy(baseline_output, output, output_size * sizeof(float));
                first = false;
                printf("\n");  // Add newline after first test
            } else if (!test.isCPU) {  // Skip CPU tests for validation
                if (baseline_output != nullptr) {
                    const float* output = std::get<2>(args_tuple);
                    // Use the actual test name for validation
                    printf("Validating %s against naive GPU baseline:\n", test.name.c_str());
                    checkResults(baseline_output, output, output_size, tolerance, test.name.c_str());
                    printf("\n");  // Add newline after validation, before next test
                }
                results.push_back(result);
            }
        }

        // Print performance summary
        printPerformanceSummary<CompareMode::BASE_ONLY>(
            test_name.c_str(), dimensions,
            results.data(), results.size(),
            baseline);

        // Cleanup
        delete[] baseline_output;
        baseline_output = nullptr;
    }

    // Get name of current test
    const char* getCurrentTestName() const {
        if (current_test < tests.size()) {
            return tests[current_test].name.c_str();
        }
        return "Unknown Test";
    }

    ~TestRegistry() {
        delete[] baseline_output;
    }
};

// Add near the top with other utility functions
__host__ __device__ inline int ceil_div(int n, int d) {
    return (n + d - 1) / d;
}