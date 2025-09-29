#include <cuda_runtime.h>
#include <stdio.h>
#include "util.h"

void sumArrays(float *a, float *b, float *res, const int size)
{
    for (int i=0; i < size; i+=4)
    {
        res[i] = a[i] + b[i];
        res[i+1] = a[i+1] + b[i+1];
        res[i+2] = a[i+2] + b[i+2];
        res[i+3] = a[i+3] + b[i+3];
    }
}

__global__ void sumArraysGPU(float *a, float *b, float *res)
{ 
    int i = threadIdx.x;
    res[i] = a[i] + b[i];
    res[i] = __hadd(res[i],res[i]);
}

__global__ void elementwise_add_fp32_kernel(float *a, float *b, float *res, int N) 
{
    
}

__global__ void checkIndex(void)
{
  printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d)\
  gridDim(%d,%d,%d)\n",threadIdx.x,threadIdx.y,threadIdx.z,
  blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,
  gridDim.x,gridDim.y,gridDim.z);
}

int main() 
{
    int dev = 0;
    cudaSetDevice(dev);
    int nElem = 20;
    int nByte = nElem * sizeof(float);

    float *a_h = (float*)malloc(nByte);
    float *b_h = (float*)malloc(nByte);
    float *res_h = (float*)malloc(nByte);
    float *res_from_gpu_h = (float*)malloc(nByte);
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    float *a_d, *b_d, *res_d;
    cudaMalloc((float**)&a_d, nByte);
    cudaMalloc((float**)&b_d,nByte);
    cudaMalloc((float**)&res_d, nByte);

    initialData(a_h, nElem);
    initialData(b_h, nElem);

    cudaMemcpy(a_d,a_h,nByte,cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b_h,nByte,cudaMemcpyHostToDevice);

    dim3 block(nElem);
    dim3 grid(nElem/block.x);
    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d);
    printf("Execution configuration<<<%d,%d>>>\n",block.x,grid.x);
    cudaMemcpy(res_from_gpu_h,res_d,nByte,cudaMemcpyDeviceToHost);

    sumArrays(a_h, b_h, res_h, nElem);

    checkResult(res_h, res_from_gpu_h, nElem);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);
    cudaFree(res_from_gpu_h);

    int num = 6;
    dim3 block_2(3);
    dim3 grid_2((num + block_2.x - 1) / block_2.x);
    printf("grid.x %d grid.y %d grid.z %d\n",grid_2.x,grid_2.y,grid_2.z);
    printf("block.x %d block.y %d block.z %d\n",block_2.x,block_2.y,grid_2.z);
    checkIndex<<<grid_2,block_2>>>();
    cudaDeviceReset();
    return 0;
}