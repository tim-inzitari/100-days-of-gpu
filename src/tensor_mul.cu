// Main compilation command that specifies:
// - nvcc: NVIDIA CUDA compiler
// - O3: Maximum optimization level
// - arch=sm_86: Target Ampere architecture
// - std=c++20: Use C++20 language features
// - use_fast_math: Enable fast math optimizations
// - Additional compiler flags for OpenMP, PIC, threading, and native architecture
// compile with:

//nvcc -O3 -arch=sm_86 -std=c++20 --use_fast_math -Xcompiler "-fopenmp -fPIC -pthread -march=native" -I. tensor_mul.cu -o tensor_mul -lcublas

//------------------------------------------------------------------------------
// This file contains multiple implementations of tensor multiplication
// Each implementation showcases different optimization techniques
//------------------------------------------------------------------------------
// Implementation list includes naive GPU, CPU with OpenMP, shared memory version,
// vectorized version, warp-optimized version, double-buffered, and tensor core
//------------------------------------------------------------------------------

// Include for standard input/output operations (printf, etc.)
#include <stdio.h>
// Include for memory allocation (malloc, free) and other standard functions
#include <stdlib.h>
// Include for string manipulation functions (memset, etc.)
#include <string.h>
// Include for mathematical functions (sqrt, fabs, etc.)
#include <math.h>
// Include for time-related functions (clock, etc.)
#include <time.h>
// Include for CUDA runtime functions (cudaMalloc, cudaMemcpy, etc.)
#include <cuda_runtime.h>
// Include for NVIDIA's Basic Linear Algebra Subroutines library
#include <cublas_v2.h>
// Include for OpenMP parallel processing directives
#include <omp.h>
// Include for quad-precision floating point arithmetic
#include <quadmath.h>
// Include for CUDA half-precision (FP16) data types
#include <cuda_fp16.h>
// Enable Tensor Float 32 (TF32) mode for tensor operations
#define WMMA_ENABLE_TF32
// Include for CUDA's Warp Matrix Multiply-Accumulate operations
#include <mma.h>
// Bring NVIDIA CUDA WMMA namespace into scope for tensor operations
using namespace nvcuda::wmma;
// Bring experimental WMMA features into scope
using namespace nvcuda::wmma::experimental;
// Include the new header
#include "gpu_utils.cuh"

//------------------------------------------------------------------------------
// Test Configuration
//------------------------------------------------------------------------------
// Type alias for test registry with tensor multiplication parameter types
using TensorTestRegistry = TestRegistry<const float*, const float*, float*, 
                                      int, int, int, int, int>;

// Create test registry with CPU testing enabled
static TensorTestRegistry tensor_tests("Tensor Multiplication", 1e-5f, false);

// Forward declare all test implementations
PerfMetrics runCPUTest(const float*, const float*, float*, int, int, int, int, int);
PerfMetrics runTestNaive(const float*, const float*, float*, int, int, int, int, int);
PerfMetrics runTestSharedMemory(const float*, const float*, float*, int, int, int, int, int);
PerfMetrics runTestCublas(const float*, const float*, float*, int, int, int, int, int);
PerfMetrics runTestVectorized(const float*, const float*, float*, int, int, int, int, int);
PerfMetrics runTestWarpOptimized(const float*, const float*, float*, int, int, int, int, int);
PerfMetrics runTestDoubleBuffered(const float*, const float*, float*, int, int, int, int, int);
PerfMetrics runTestTensorCore(const float*, const float*, float*, int, int, int, int, int);

// Initialize all tests
void initializeTests() {
    tensor_tests.addTest("CPU (OpenMP)", runCPUTest, false, true);
    tensor_tests.addTest("Naive GPU", runTestNaive, true);
    tensor_tests.addTest("Shared Memory", runTestSharedMemory, true);
    tensor_tests.addTest("cuBLAS", runTestCublas, true);
    tensor_tests.addTest("Vectorized", runTestVectorized, true);
    tensor_tests.addTest("Warp-Optimized", runTestWarpOptimized, true);
    tensor_tests.addTest("Double-Buffered", runTestDoubleBuffered, true);
    tensor_tests.addTest("Tensor Core", runTestTensorCore, true);
}

//------------------------------------------------------------------------------
// Global constants used throughout the program
//------------------------------------------------------------------------------
// Define size of shared memory tiles for matrix multiplication
#define TILE_SIZE      32
// Define size of thread blocks for kernel launches
#define BLOCK_SIZE     32
// Macro to compute minimum of two values (used for boundary checking)
#define MIN(a,b)       ((a) < (b) ? (a) : (b))

// CPU blocking parameters for cache-efficient implementation
#define BLOCK_SIZE_M   32    // Block size for matrix rows (cache line optimization)
#define BLOCK_SIZE_N   32    // Block size for inner dimension (register optimization)
#define BLOCK_SIZE_L   32    // Block size for matrix columns (cache line optimization)

//------------------------------------------------------------------------------
// File header explaining this is a refactored version containing multiple matrix multiplication implementations
//------------------------------------------------------------------------------
// Tests included: naive GPU, CPU (OpenMP), shared memory, vectorized, warp-optimized, double-buffered, tensor core
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Forward declarations of GPU kernel functions - these declare the interfaces
// for all tensor multiplication implementations that will be defined later
//------------------------------------------------------------------------------

// Test 0: Basic GPU implementation with no optimizations
// Parameters:
//   A, B: Input matrices stored as linear arrays
//   C: Output matrix
//   batch_size: Number of matrix multiplications to perform
//   m, n, k, l: Matrix dimensions where A is [m x n] and B is [k x l]
__global__ void tensor_mul(const float* A, const float* B, float* C,
                           int batch_size, int m, int n, int k, int l);

// Test 2: Implementation using shared memory to reduce global memory access
// Uses TILE_SIZE x TILE_SIZE shared memory blocks to cache matrix elements
// Same parameters as tensor_mul
__global__ void tensor_mul_shared(const float* A, const float* B, float* C,
                                  int batch_size, int m, int n, int k, int l);

// Test 4: Vectorized implementation that processes multiple elements per thread
// Uses regular floats but organizes computation to enable compiler vectorization
// Same parameters as tensor_mul
__global__ void tensor_mul_vectorized(const float* A, const float* B, float* C,
                                      int batch_size, int m, int n, int k, int l);

// Test 5: Implementation optimized for warp-level operations
// Organizes threads within warps for efficient memory access and computation
// Same parameters as tensor_mul
__global__ void tensor_mul_warp_optimized(const float* A, const float* B, float* C,
                                          int batch_size, int m, int n, int k, int l);

// Test 6: Double-buffered implementation for overlapping computation and memory access
// Uses __restrict__ keyword to enable better compiler optimizations
// Same parameters as tensor_mul
__global__ void tensor_mul_double_buffered(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float*       __restrict__ C,
                                           int batch_size, int m, int n, int k, int l);

// Test 7: Tensor Core Implementation using TF32 precision on NVIDIA Ampere GPUs
// This kernel leverages Tensor Cores for high-performance matrix multiplication
// Parameters:
//   A, B: Input matrices in float precision
//   C: Output matrix
//   batch_size: Number of matrix multiplications to perform
//   M, N, k, L: Matrix dimensions where A is [M x N] and B is [k x L]
__global__ void tensor_mul_tensorcore(const float* A, const float* B, float* C,
                                     int batch_size, int M, int N, int k, int L);

// Test 7 variant: Tensor Core implementation optimized for lower numerical error
// Uses float precision WMMA operations where possible
// Same parameters as tensor_mul_tensorcore
__global__ void tensor_mul_tensorcore_float(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float*       __restrict__ C,
                                            int batch_size, int M, int N, int K, int L);

//------------------------------------------------------------------------------
// Utility function: Initialize matrices A and B with random float values
// Parameters:
//   A, B: Matrices to initialize
//   batch_size: Number of matrix pairs to initialize
//   m, n, k, l: Matrix dimensions
//------------------------------------------------------------------------------
void initMatrices(float* A, float* B, int batch_size, int m, int n, int k, int l) {
    // Iterate over each batch
    for (int b = 0; b < batch_size; b++){
        // Initialize matrix A for this batch
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                // Calculate linear index for 3D array access
                // Generate random float between 0 and 1
                A[b * m * n + i * n + j] = (float)(rand() % 100) / 100.0f;
            }
        }
        // Initialize matrix B for this batch
        for (int i = 0; i < k; i++){
            for (int j = 0; j < l; j++){
                // Calculate linear index for 3D array access
                // Generate random float between 0 and 1
                B[b * k * l + i * l + j] = (float)(rand() % 100) / 100.0f;
            }
        }
    }
}

//------------------------------------------------------------------------------
// CPU Implementation using OpenMP and __float128 (quad precision) for improved accuracy
// Parameters:
//   A, B: Input matrices
//   C: Output matrix
//   batch_size: Number of matrix multiplications to perform
//   m, n, k, l: Matrix dimensions
//------------------------------------------------------------------------------
void cpu_matrix_multiply(const float* A, const float* B, float* C,
                         int batch_size, int m, int n, int k, int l) {
    // Initialize output matrix C to zero
    memset(C, 0, batch_size * m * l * sizeof(float));
    
    // Parallelize three nested loops using OpenMP
    // collapse(3) combines three loops into one parallel region
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; b++){
        // Process matrix blocks for cache efficiency
        for (int i_start = 0; i_start < m; i_start += BLOCK_SIZE_M){
            for (int j_start = 0; j_start < l; j_start += BLOCK_SIZE_L){
                // Calculate actual block dimensions (handle edge cases)
                int imax = MIN(i_start + BLOCK_SIZE_M, m);
                int jmax = MIN(j_start + BLOCK_SIZE_L, l);
                
                // Process elements within current block
                for (int i = i_start; i < imax; i++){
                    for (int j = j_start; j < jmax; j++){
                        // Use quad precision accumulator for higher accuracy
                        __float128 sum = (__float128)0.0;
                        
                        // Calculate base indices for current position
                        size_t base_a = (size_t)b * m * n + (size_t)i * n;
                        size_t base_b = (size_t)b * k * l + (size_t)j;
                        
                        // Compute dot product for current element
                        for (int p = 0; p < n; p++){
                            // Load values from A and B matrices
                            __float128 a_val = A[base_a + p];
                            __float128 b_val = B[base_b + p * l];
                            // Accumulate product in quad precision
                            sum += a_val * b_val;
                        }
                        
                        // Calculate final output index
                        size_t idx = (size_t)b * m * l + (size_t)i * l + j;
                        // Atomically add result to output matrix (thread-safe)
                        #pragma omp atomic
                        C[idx] += (float)sum;
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Test 0: Run naive GPU implementation
// Parameters:
//   h_A, h_B: Input matrices
//   h_C: Output matrix
//   batch_size, m, n, k, l: Matrix dimensions
// Returns: Performance metrics for this implementation
//------------------------------------------------------------------------------
PerfMetrics runTestNaive(const float* h_A, const float* h_B, float* h_C,
                         int batch_size, int m, int n, int k, int l) {
    dim3 block(32, 32);
    dim3 grid(ceil_div(m, 32), ceil_div(l, 32), batch_size);
    LaunchConfig config(grid, block);  // Create launch config
    
    size_t size_A = batch_size * m * n;
    size_t size_B = batch_size * k * l;
    size_t size_C = batch_size * m * l;
    size_t flops_per_thread = 2 * n;
    
    return runGpuTest<float>(
        tensor_tests.getCurrentTestName(),
        tensor_mul,
        h_A, h_B, h_C,
        size_A, size_B, size_C,
        config,  // Pass launch config instead of separate grid and block
        flops_per_thread,
        batch_size, m, n, k, l
    );
}

//------------------------------------------------------------------------------
// Test 2: Run shared memory implementation
// Parameters:
//   h_A, h_B: Input matrices
//   h_C: Output matrix
//   batch_size, m, n, k, l: Matrix dimensions
// Returns: Performance metrics for this implementation
//------------------------------------------------------------------------------
PerfMetrics runTestSharedMemory(const float* h_A, const float* h_B, float* h_C,
                                int batch_size, int m, int n, int k, int l) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(ceil_div(m, TILE_SIZE), ceil_div(l, TILE_SIZE), batch_size);
    
    // Calculate shared memory size
    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    LaunchConfig config(grid, block, shared_mem_size);
    
    size_t size_A = batch_size * m * n;
    size_t size_B = batch_size * k * l;
    size_t size_C = batch_size * m * l;
    size_t flops_per_thread = 2 * n;
    
    return runGpuTest<float>(
        tensor_tests.getCurrentTestName(),
        tensor_mul_shared,
        h_A, h_B, h_C,
        size_A, size_B, size_C,
        config,
        flops_per_thread,
        batch_size, m, n, k, l
    );
}

//------------------------------------------------------------------------------
// Test 3: cuBLAS Implementation (Placeholder)
//------------------------------------------------------------------------------
PerfMetrics runTestCublas(const float* h_A, const float* h_B, float* h_C,
                         int batch_size, int m, int n, int k, int l) {
    PerfMetrics pm = {0};
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, batch_size * m * n * sizeof(float));
    cudaMalloc((void**)&d_B, batch_size * n * l * sizeof(float));
    cudaMalloc((void**)&d_C, batch_size * m * l * sizeof(float));

    // Time H2D transfers
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, batch_size * m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, batch_size * n * l * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.transferTime = elapsed;

    float alpha = 1.0f, beta = 0.0f;
    // cuBLAS uses column-major ordering. This call transposes the operation so that
    // our row-major data is correctly multiplied.
    int lda = n, ldb = l, ldc = l;  // Leading dimensions for column-major format
    long long strideA = m * n;
    long long strideB = n * l;
    long long strideC = m * l;

    cudaEventRecord(start);

    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              l, m, n,  // Swap m and l for column-major
                              &alpha,
                              d_B, ldb, strideB,  // Swap A and B
                              d_A, lda, strideA,
                              &beta,
                              d_C, ldc, strideC, batch_size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.kernelTime = elapsed;

    // Time D2H transfer
    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, batch_size * m * l * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.d2hTime = elapsed;

    pm.totalTime = pm.transferTime + pm.kernelTime + pm.d2hTime;
    pm.gflops = (2.0f * batch_size * m * n * l) / (pm.kernelTime * 1e6f);

    printf("%s:\n", tensor_tests.getCurrentTestName());
    printf("   H2D: %.3f ms, Kernel: %.3f ms, D2H: %.3f ms, Total: %.3f ms, GFLOPS: %.2f\n",
           pm.transferTime, pm.kernelTime, pm.d2hTime, pm.totalTime, pm.gflops);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return pm;
}

//------------------------------------------------------------------------------
// Test 4: Vectorized Implementation (using regular floats)
//------------------------------------------------------------------------------
PerfMetrics runTestVectorized(const float* h_A, const float* h_B, float* h_C,
                              int batch_size, int m, int n, int k, int l) {
    dim3 block(32, 32);
    dim3 grid(ceil_div(m, 32), ceil_div(l, 32), batch_size);
    LaunchConfig config(grid, block);
    
    size_t size_A = batch_size * m * n;
    size_t size_B = batch_size * k * l;
    size_t size_C = batch_size * m * l;
    size_t flops_per_thread = 2 * n;
    
    return runGpuTest<float>(
        tensor_tests.getCurrentTestName(),
        tensor_mul_vectorized,
        h_A, h_B, h_C,
        size_A, size_B, size_C,
        config,
        flops_per_thread,
        batch_size, m, n, k, l
    );
}

//------------------------------------------------------------------------------
// Test 5: Warp-Optimized Implementation
//------------------------------------------------------------------------------
PerfMetrics runTestWarpOptimized(const float* h_A, const float* h_B, float* h_C,
                                 int batch_size, int m, int n, int k, int l) {
    dim3 block(256);  // 8 warps per block
    dim3 grid(ceil_div(m, 8), ceil_div(l, 32), batch_size);
    LaunchConfig config(grid, block);
    
    size_t size_A = batch_size * m * n;
    size_t size_B = batch_size * k * l;
    size_t size_C = batch_size * m * l;
    size_t flops_per_thread = 2 * n;
    
    return runGpuTest<float>(
        tensor_tests.getCurrentTestName(),
        tensor_mul_warp_optimized,
        h_A, h_B, h_C,
        size_A, size_B, size_C,
        config,
        flops_per_thread,
        batch_size, m, n, k, l
    );
}

//------------------------------------------------------------------------------
// Test 6: Double-Buffered Implementation (Rewritten for near-zero error)
//------------------------------------------------------------------------------
PerfMetrics runTestDoubleBuffered(const float* h_A, const float* h_B, float* h_C,
                                  int batch_size, int m, int n, int k, int l) {
    dim3 block(32, 32);
    dim3 grid(ceil_div(m, 32), ceil_div(l, 32), batch_size);
    LaunchConfig config(grid, block);
    
    size_t size_A = batch_size * m * n;
    size_t size_B = batch_size * k * l;
    size_t size_C = batch_size * m * l;
    size_t flops_per_thread = 2 * n;
    
    return runGpuTest<float>(
        tensor_tests.getCurrentTestName(),
        tensor_mul_double_buffered,
        h_A, h_B, h_C,
        size_A, size_B, size_C,
        config,
        flops_per_thread,
        batch_size, m, n, k, l
    );
}

//------------------------------------------------------------------------------
// Test 7: Tensor Core Implementation using TF32 precision on NVIDIA Ampere GPUs
// This kernel leverages Tensor Cores for high-performance matrix multiplication
// Parameters:
//   A, B: Input matrices in float precision
//   C: Output matrix
//   batch_size: Number of matrix multiplications to perform
//   M, N, k, L: Matrix dimensions where A is [M x N] and B is [k x L]
//------------------------------------------------------------------------------
PerfMetrics runTestTensorCore(const float* h_A, const float* h_B, float* h_C,
                              int batch_size, int m, int n, int k, int l) {
    if (m % 16 != 0 || n % 16 != 0 || l % 16 != 0) {
        printf("Test 7: Tensor Core Implementation: Skipped (dimensions must be multiples of 16).\n");
        return PerfMetrics{0};
    }
    
    dim3 block(128, 1);  // 4 warps per block
    dim3 grid(ceil_div(m, 64), ceil_div(l, 64), batch_size);
    
    // Calculate shared memory size for partial tile handling
    size_t shared_mem_size = 16 * 16 * sizeof(float);
    LaunchConfig config(grid, block, shared_mem_size);
    
    size_t size_A = batch_size * m * n;
    size_t size_B = batch_size * k * l;
    size_t size_C = batch_size * m * l;
    size_t flops_per_thread = 2 * n;
    
    return runGpuTest<float>(
        tensor_tests.getCurrentTestName(),
        tensor_mul_tensorcore,
        h_A, h_B, h_C,
        size_A, size_B, size_C,
        config,
        flops_per_thread,
        batch_size, m, n, k, l
    );
}

//------------------------------------------------------------------------------
// Test 7: Tensor Core Implementation (Rewritten for low error w/ float WMMA)
//------------------------------------------------------------------------------
__global__ void tensor_mul_tensorcore_float(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float*       __restrict__ C,
                                            int batch_size, int M, int N, int K, int L)
{
#if (__CUDA_ARCH__ < 800)
    // On older GPUs, you won't have TF32 or float WMMA support. 
    // You can either do nothing or call a fallback kernel here.
    return;
#endif

    // Each block handles one 16×16 tile of C.
    int warpM = blockIdx.x; 
    int warpN = blockIdx.y; 
    int batch = blockIdx.z;
    if (batch >= batch_size) return;

    // Create fragments (float or TF32 if needed)
    nvcuda::wmma::fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    nvcuda::wmma::fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    nvcuda::wmma::fragment<accumulator, 16, 16, 16, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    // Traverse the K dimension in chunks of 16
    for (int k_step = 0; k_step < N; k_step += 16)
    {
        int loadK = (k_step + 16 <= N) ? 16 : (N - k_step);
        if (loadK <= 0) break;

        // Global pointers for A, B
        float* a_ptr = const_cast<float*>(A + (batch * M * N) + (warpM * 16 * N) + k_step);
        float* b_ptr = const_cast<float*>(B + (batch * N * L) + (k_step * L) + (warpN * 16));

        // Convert float to half before loading
        __half *a_half = (__half*)malloc(16 * N * sizeof(__half));
        __half *b_half = (__half*)malloc(16 * L * sizeof(__half));
        
        // Convert the float data to half
        for(int i = 0; i < 16; i++) {
            for(int j = 0; j < N; j++) {
                a_half[i * N + j] = __float2half(a_ptr[k_step + i * N + j]);
            }
        }
        for(int i = 0; i < 16; i++) {
            for(int j = 0; j < L; j++) {
                b_half[i * L + j] = __float2half(b_ptr[(k_step + i) * L + j]);
            }
        }
        
        // Load the converted half-precision data
        nvcuda::wmma::load_matrix_sync(a_frag, a_half, N);
        nvcuda::wmma::load_matrix_sync(b_frag, b_half, L);
        
        free(a_half);
        free(b_half);

        // Multiply-Accumulate
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Write output tile
    float* outC = C + (batch * M * L) + (warpM * 16 * L) + (warpN * 16);

    // Boundary checks
    int rowsLeft = M - warpM * 16;
    int colsLeft = L - warpN * 16;
    if (rowsLeft >= 16 && colsLeft >= 16)
    {
        // Full tile
        store_matrix_sync(outC, acc_frag, L, mem_row_major);
    }
    else
    {
        // Partial tile -> store to shared memory first
        __shared__ float shm[16 * 16];
        store_matrix_sync(shm, acc_frag, 16, mem_row_major);
        __syncthreads();

        // Only threads in [0..15,0..15] handle copying
        int tx = threadIdx.x; 
        int ty = threadIdx.y;
        if (tx < 16 && ty < 16)
        {
            if (tx < rowsLeft && ty < colsLeft)
            {
                outC[tx * L + ty] = shm[tx * 16 + ty];
            }
        }
    }
}

// Test 0: Naive GPU kernel implementation
__global__ void tensor_mul(const float* A, const float* B, float* C,
                           int batch_size, int m, int n, int k, int l) {
    int batch = blockIdx.z;
    int row   = blockIdx.x * blockDim.x + threadIdx.x;
    int col   = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch < batch_size && row < m && col < l) {
        float sum = 0.0f;
        for (int p = 0; p < n; p++) {
            sum += A[batch * m * n + row * n + p] *
                   B[batch * k * l + p * l + col];
        }
        C[batch * m * l + row * l + col] = sum;
    }
}

// Device helper function to perform naive multiplication.
// This function can be called from inside __global__ kernels.
__device__ void tensor_mul_device(const float* A, const float* B, float* C,
                                  int batch_size, int m, int n, int k, int l) {
    int batch = blockIdx.z;
    int row   = blockIdx.x * blockDim.x + threadIdx.x;
    int col   = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch < batch_size && row < m && col < l) {
        float sum = 0.0f;
        for (int p = 0; p < n; p++) {
            sum += A[batch * m * n + row * n + p] *
                   B[batch * k * l + p * l + col];
        }
        C[batch * m * l + row * l + col] = sum;
    }
}

// Revised Shared Memory Kernel
__global__ void tensor_mul_shared(const float* A, const float* B, float* C,
                                  int batch_size, int m, int n, int k, int l) {
    // Each batch computes: (m x n) * (n x l) = (m x l)
    int batch = blockIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // row in C and A
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // col in C and B

    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    const int numTiles = ceil_div(n, TILE_SIZE);  // Instead of (n + TILE_SIZE - 1) / TILE_SIZE

    // Precompute batch offsets.
    int batch_offset_A = batch * m * n;
    int batch_offset_B = batch * n * l;

    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_SIZE + threadIdx.y;   // Column for tile from A.
        int tiledRow = t * TILE_SIZE + threadIdx.x;    // Row for tile from B.

        // Load A tile
        if (row < m && tiledCol < n)
            A_shared[threadIdx.x][threadIdx.y] = A[batch_offset_A + row * n + tiledCol];
        else
            A_shared[threadIdx.x][threadIdx.y] = 0.0f;

        // Load B tile
        if (tiledRow < n && col < l)
            B_shared[threadIdx.x][threadIdx.y] = B[batch_offset_B + tiledRow * l + col];
        else
            B_shared[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        // Multiply the two tiles
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_shared[threadIdx.x][i] * B_shared[i][threadIdx.y];
        }

        __syncthreads();
    }

    // Write back the result in global memory.
    if (row < m && col < l) {
        C[batch * m * l + row * l + col] = sum;
    }
}

// Test 4: Vectorized Implementation (using regular floats)
// This kernel processes multiple elements per thread to improve throughput
// Parameters:
//   A, B: Input matrices in float precision
//   C: Output matrix
//   batch_size: Number of matrix multiplications to perform
//   m, n, k, l: Matrix dimensions where A is [m x n] and B is [k x l]
__global__ void tensor_mul_vectorized(const float* A, const float* B, float* C,
                                      int batch_size, int m, int n, int k, int l) {
    // Calculate global thread indices
    int batch = blockIdx.z;                              // Current batch
    int row   = blockIdx.x * blockDim.x + threadIdx.x;  // Global row index
    int col = blockIdx.y * blockDim.y + threadIdx.y;    // Global column index
    
    // Only process if thread maps to valid matrix elements
    if (batch < batch_size && row < m && col < l) {
        float sum = 0.0f;
        
        // Vectorized loop: process 4 elements at a time for better throughput
        // This allows the compiler to use SIMD instructions where available
        int p;
        #pragma unroll  // Hint to compiler to unroll this loop
        for (p = 0; p < n-3; p += 4) {
            // Compute 4 multiply-adds in one iteration
            // This reduces loop overhead and enables instruction-level parallelism
            sum += A[batch * m * n + row * n + p] * B[batch * k * l + p * l + col]
                 + A[batch * m * n + row * n + p + 1] * B[batch * k * l + (p + 1) * l + col]
                 + A[batch * m * n + row * n + p + 2] * B[batch * k * l + (p + 2) * l + col]
                 + A[batch * m * n + row * n + p + 3] * B[batch * k * l + (p + 3) * l + col];
        }
        
        // Handle remaining elements that don't fit in vector of 4
        for (; p < n; p++) {
            sum += A[batch * m * n + row * n + p] * B[batch * k * l + p * l + col];
        }
        
        // Write final result to global memory
        C[batch * m * l + row * l + col] = sum;
    }
}

// Test 5: Warp-Optimized Implementation
// This kernel organizes computation to maximize warp efficiency
// Uses warp-level parallelism for better memory coalescing and execution efficiency
// Parameters:
//   A, B: Input matrices in float precision
//   C: Output matrix
//   batch_size: Number of matrix multiplications to perform
//   m, n, k, l: Matrix dimensions where A is [m x n] and B is [k x l]
__global__ void tensor_mul_warp_optimized(const float* A, const float* B, float* C,
                                          int batch_size, int m, int n, int k, int l) {
    // Calculate warp-specific indices
    int batch = blockIdx.z;                          // Current batch
    int warp_id = threadIdx.x / 32;                  // Warp ID within block (32 threads = 1 warp)
    int lane = threadIdx.x % 32;                     // Thread's lane within its warp
    int row = blockIdx.x * (blockDim.x/32) + warp_id;// Global row index
    int col = blockIdx.y * 32 + lane;                // Global column index (one per lane)
    
    // Only process if thread maps to valid matrix elements
    if (batch < batch_size && row < m && col < l) {
        // Each thread accumulates one element of the output matrix
        float sum = 0.0f;
        
        // Calculate base pointers for coalesced memory access
        // All threads in a warp access consecutive memory locations
        const float* batch_A = A + batch * m * n + row * n;  // Row of matrix A
        const float* batch_B = B + batch * k * l + col;      // Column of matrix B
        
        // Process the reduction in chunks for better instruction-level parallelism
        #pragma unroll 4  // Hint to compiler to unroll loop 4 times
        for (int p = 0; p < n; p++) {
            // Coalesced read from A (all threads in warp read consecutive elements)
            // Broadcast read from B (all threads read same element)
            sum += batch_A[p] * batch_B[p * l];
        }
        
        // Write result directly - memory access is coalesced across the warp
        // No need for warp reduction since each thread computes one output element
        C[batch * m * l + row * l + col] = sum;
    }
}

// Test 6: Double-Buffered Implementation (Rewritten for near-zero error)
// This kernel uses shared memory tiling with double buffering to overlap 
// computation with memory access, while maintaining high numerical accuracy
// Parameters:
//   A, B: Input matrices (with __restrict__ to help compiler optimization)
//   C: Output matrix
//   batch_size: Number of matrix multiplications to perform
//   m, n, k, l: Matrix dimensions where A is [m x n] and B is [k x l]
__global__ void tensor_mul_double_buffered(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float*       __restrict__ C,
                                           int batch_size, int m, int n, int k, int l) {
    // Early exit for threads beyond batch size
    int batch = blockIdx.z;
    if (batch >= batch_size) return;

    // Define tile size for shared memory blocking
    // 32x32 tiles provide good balance between occupancy and cache utilization
    constexpr int TILE_DIM = 32;
    
    // Calculate global thread positions
    const int row = blockIdx.x * TILE_DIM + threadIdx.x;  // Global row index
    const int col = blockIdx.y * TILE_DIM + threadIdx.y;  // Global column index
    
    // Declare shared memory tiles for A and B matrices
    // These will be used to cache data from global memory
    __shared__ float As[TILE_DIM][TILE_DIM];  // Tile from matrix A
    __shared__ float Bs[TILE_DIM][TILE_DIM];  // Tile from matrix B
    
    // Initialize accumulator for dot product
    float sum = 0.0f;
    
    // Calculate base pointers for current batch
    // This avoids repeated address calculations in the loop
    const float* batch_A = A + batch * m * n;  // Point to start of current batch in A
    const float* batch_B = B + batch * k * l;  // Point to start of current batch in B
    
    // Main loop over tiles
    // Process the matrices in TILE_DIM x TILE_DIM blocks
    for (int tile = 0; tile < n; tile += TILE_DIM) {
        // Collaborative loading of tiles into shared memory
        // Each thread loads one element of A and one element of B
        if (row < m && (tile + threadIdx.y) < n) {
            // Load element from A if within matrix bounds
            As[threadIdx.x][threadIdx.y] = batch_A[row * n + tile + threadIdx.y];
        }
        if ((tile + threadIdx.x) < n && col < l) {
            // Load element from B if within matrix bounds
            Bs[threadIdx.x][threadIdx.y] = batch_B[(tile + threadIdx.x) * l + col];
        }
        
        // Ensure all threads have loaded their data before computation
        __syncthreads();
        
        // Compute partial dot product for this tile
        if (row < m && col < l) {
            // Unroll loop for better instruction-level parallelism
            #pragma unroll
            for (int k = 0; k < TILE_DIM; k++) {
                // Only accumulate if within the input matrix bounds
                if (tile + k < n) {
                    sum += As[threadIdx.x][k] * Bs[k][threadIdx.y];
                }
            }
        }
        
        // Ensure all threads are done with shared memory before next iteration
        __syncthreads();
    }
    
    // Write final result to global memory
    // Only write if thread's indices are within matrix bounds
    if (row < m && col < l) {
        C[batch * m * l + row * l + col] = sum;
    }
}

// Test 7: Tensor Core Implementation using TF32 precision on NVIDIA Ampere GPUs
// This kernel leverages Tensor Cores for high-performance matrix multiplication
// Parameters:
//   A, B: Input matrices in float precision
//   C: Output matrix
//   batch_size: Number of matrix multiplications to perform
//   M, N, k, L: Matrix dimensions where A is [M x N] and B is [k x L]
__global__ void tensor_mul_tensorcore(const float* A, const float* B, float* C,
                                     int batch_size, int M, int N, int k, int L) {
    // Only compile this code for Ampere (SM80+) architectures that support TF32
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    // Declare shared memory for temporary storage during partial tile handling
    extern __shared__ float shmem[];
    
    // Calculate warp and batch indices from block coordinates
    const int warpM = blockIdx.x;    // Warp's row position in output matrix
    const int warpN = blockIdx.y;    // Warp's column position in output matrix
    const int batch = blockIdx.z;    // Current batch being processed
    
    // Calculate pointers to current 16x16 tile in input/output matrices
    // Each warp processes a 16x16 tile (Tensor Core requirement)
    const float* a_ptr = A + batch * M * N + warpM * 16 * N;  // Pointer to current A tile
    const float* b_ptr = B + batch * N * L + warpN * 16;      // Pointer to current B tile
    float* c_ptr = C + batch * M * L + warpM * 16 * L + warpN * 16;  // Pointer to output tile
    
    // Declare WMMA fragments for matrix multiplication
    // These are special registers used by Tensor Cores
    // matrix_a: Input matrix A fragment (16x16x16)
    // matrix_b: Input matrix B fragment (16x16x16)
    // accumulator: Output matrix C fragment (16x16)
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
    
    // Initialize accumulator fragment to zeros
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
    
    // Main computation loop - process input matrices in 16x16 tiles
    for (int k_step = 0; k_step < N; k_step += 16) {
        if (k_step + 16 <= N) {  // Check if we have a full tile
            // Allocate temporary storage for FP16 conversion
            // Note: Using malloc in device code is not recommended for performance
            // Better to pre-allocate in shared memory
            __half *a_half = (__half*)malloc(16 * N * sizeof(__half));
            __half *b_half = (__half*)malloc(16 * L * sizeof(__half));
            
            // Convert matrix A tile from float to half precision
            for(int i = 0; i < 16; i++) {
                for(int j = 0; j < N; j++) {
                    a_half[i * N + j] = __float2half(a_ptr[k_step + i * N + j]);
                }
            }
            
            // Convert matrix B tile from float to half precision
            for(int i = 0; i < 16; i++) {
                for(int j = 0; j < L; j++) {
                    b_half[i * L + j] = __float2half(b_ptr[(k_step + i) * L + j]);
                }
            }
            
            // Load converted data into WMMA fragments
            nvcuda::wmma::load_matrix_sync(a_frag, a_half, N);  // Load A fragment
            nvcuda::wmma::load_matrix_sync(b_frag, b_half, L);  // Load B fragment
            
            // Free temporary conversion buffers
            free(a_half);
            free(b_half);

            // Perform matrix multiplication using Tensor Cores
            // This operation is done in TF32 precision on Ampere GPUs
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // Calculate remaining rows and columns for edge cases
    const int rows_left = max(0, M - warpM * 16);  // Remaining rows in last tile
    const int cols_left = max(0, L - warpN * 16);  // Remaining columns in last tile
    
    // Handle output storage based on tile completeness
    if (rows_left >= 16 && cols_left >= 16) {
        // For complete tiles, store directly to output
        nvcuda::wmma::store_matrix_sync(c_ptr, acc_frag, L, nvcuda::wmma::mem_row_major);
    } else {
        // For partial tiles, store to shared memory first
        float* temp_storage = (float*)shmem;
        nvcuda::wmma::store_matrix_sync(temp_storage, acc_frag, 16, nvcuda::wmma::mem_row_major);
        __syncthreads();  // Ensure all threads have written to shared memory
        
        // Copy valid elements to output (handling edge cases)
        if (threadIdx.x < 16 && threadIdx.y < 16) {
            if (threadIdx.x < rows_left && threadIdx.y < cols_left) {
                c_ptr[threadIdx.x * L + threadIdx.y] = temp_storage[threadIdx.x * 16 + threadIdx.y];
            }
        }
    }
#endif
}

// Wrap CPU implementation in same interface as GPU tests
PerfMetrics runCPUTest(const float* h_A, const float* h_B, float* h_C,
                       int batch_size, int m, int n, int k, int l) {
    PerfMetrics pm = {0};
    
    double start = omp_get_wtime();
    cpu_matrix_multiply(h_A, h_B, h_C, batch_size, m, n, k, l);
    double end = omp_get_wtime();
    
    pm.totalTime = (end - start) * 1000.0;  // Convert to ms
    pm.gflops = (2.0 * batch_size * m * n * l) / (pm.totalTime * 1e6);
    
    return pm;
}

//------------------------------------------------------------------------------
// Main: Orchestrate all tests and print a summary.
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Validate command line arguments
    if (argc != 6) {
        printf("Usage: ./tensor_mul <batch_size> <m> <n> <k> <l>\n");
        return 1;
    }

    // Parse dimensions
    int batch_size = atoi(argv[1]);
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    int k = atoi(argv[4]);
    int l = atoi(argv[5]);
    if (batch_size <= 0 || m <= 0 || n <= 0 || k <= 0 || l <= 0) {
        printf("Error: All dimensions must be positive integers.\n");
        return 1;
    }
    
    // Calculate total elements for each matrix
    size_t total_elements_A = batch_size * m * n;
    size_t total_elements_B = batch_size * k * l;
    size_t total_elements_C = batch_size * m * l;
    
    // Allocate host matrices
    float *h_A = (float*)malloc(total_elements_A * sizeof(float));
    float *h_B = (float*)malloc(total_elements_B * sizeof(float));
    float *h_C_baseline = (float*)malloc(total_elements_C * sizeof(float));
    float *h_C_temp = (float*)malloc(total_elements_C * sizeof(float));
    if (!h_A || !h_B || !h_C_baseline || !h_C_temp) {
        printf("Error: Host memory allocation failed.\n");
        return 1;
    }
    
    // Initialize input matrices
    srand(time(NULL));
    initMatrices(h_A, h_B, batch_size, m, n, k, l);
    
    initializeTests();
    
    char dimensions[256];
    snprintf(dimensions, sizeof(dimensions), 
             "Tensor Dimensions: [%d × %d × %d] × [%d × %d × %d]",
             batch_size, m, n, batch_size, k, l);

    // Run all tests
    tensor_tests.runAll(dimensions, h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_baseline);
    free(h_C_temp);
    cudaFree(0);  // Reset CUDA context
    
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
