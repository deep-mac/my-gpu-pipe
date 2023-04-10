#define cudaErrCheck(stat)                         \
    {                                              \
        cudaErrCheck_((stat), __FILE__, __LINE__); \
    }

void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
    if (stat != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}


float cuda_time_kernel_ms(std::function<void(void)> func) {
    float time_ms;
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));

    cudaErrCheck(cudaEventRecord(start));
    func();
    cudaErrCheck(cudaGetLastError());
    cudaErrCheck(cudaEventRecord(stop));

    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&time_ms, start, stop));

    return time_ms;
}
