// Hi
#include <stdio.h>
#include <cuda.h>
#include <time.h>

//#define DEBUG


__global__
void mat_vec_mul(float *A, float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    A[i] = 0;   
    for (int k = 0; k < n; k++)
    {
        A[i] += B[i * n + k] * C[k];
    }
}


int main(int argc, char **argv)
{
    int n;
    float elapsed_time;
    cudaEvent_t start, stop;
    clock_t cpu_start, cpu_end;

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    if (argc != 2){
        printf("Usage: ./a.out <n>\n");
        return 1;
    }

    n = atoi(argv[1]);

    // Print problem size
    printf("Matrix size: %d x %d\n", n, n);
    printf("Vector size: %d\n", n);
    printf("----------------------------------------\n");

    // Allocate memory on host
    h_A = (float *)malloc(n * sizeof(float));
    h_B = (float *)malloc(n * n * sizeof(float));
    h_C = (float *)malloc(n * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < n; i++)
    {
        h_C[i] = 1;
        for (int j = 0; j < n; j++)
        {
            h_B[i * n + j] = 1;
        }
    }

    // Allocate memory on device
    cudaMalloc((void **)&d_A, n * sizeof(float));
    cudaMalloc((void **)&d_B, n * n * sizeof(float));
    cudaMalloc((void **)&d_C, n * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_C, h_C, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cpu_start = clock();
    cudaEventRecord(start);

    // Launch the kernel
    dim3 dimBlock(256);
    dim3 dimGrid(ceil_div(n, dimBlock.x));
    mat_vec_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cpu_end = clock();

    // Copy device memory to host memory
    cudaMemcpy(h_A, d_A, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print timing information
    printf("----------------------------------------\n");
    printf("Performance Results:\n");
    printf("GPU Kernel Time: %f ms\n", elapsed_time);
    printf("Total CPU Time: %f ms\n", 1000.0 * (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC);

#ifdef DEBUG
    // Print the result
    for (int i = 0; i < n; i++)
    {
        printf("%f ", h_A[i]);
    }
    printf("\n");
#endif

    // Cleanup timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}