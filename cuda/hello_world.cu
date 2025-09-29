#include <stdio.h>

__global__ void hello_world(void)
{
    printf("Hello World from GPU!\n");
}

int main()
{
    printf("CPU: Hello world!");
    hello_world<<<1,10>>>();
    cudaDeviceReset();
    return 0;
}