#pragma once
#include <iostream>
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "HelpMethods.h"

#define MUL_USING_TILES true
#define TILE_WIDTH 2
#define USE_SHARED_MEMORY true

using namespace std;

#pragma region Device and Global Methods

__global__ void add_m(int * a, int * b, int *c, int width)
{
    int i = threadIdx.x * blockDim.x * blockIdx.x;
    int j = threadIdx.y * blockDim.y * blockIdx.y;
    int index = i * width + j;

    c[index] = a[index] + b[index];
}

__global__ void sub_m(int* a, int* b, int* c, int width)
{
    int i = threadIdx.x * blockDim.x * blockIdx.x;
    int j = threadIdx.y * blockDim.y * blockIdx.y;
    int index = i * width + j;

    c[index] = a[index] - b[index];
}

__global__ void mul_m(int* a, int* b, int* c, int width)
{
    int sum = 0;
    for (int k = 0; k < width; k++)
    {
        int a_v = a[threadIdx.x * width + k];
        int b_v = b[k * width + threadIdx.y];
        sum += (a_v * b_v);
    }

    c[threadIdx.x * width + threadIdx.y] = sum;
}

__global__ void mul_m_tiles(int* a, int* b, int* c, int width)
{
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    int sum = 0;
    for (int k = 0; k < width; k++)
    {
        int a_v = a[row * width + k];
        int b_v = a[k * width + col];
        sum += (a_v * b_v);
    }

    c[row * width + col] = sum;
}

__global__ void mul_m_shared(int* a, int* b, int* c, int width)
{
    __shared__ int A[TILE_WIDTH][TILE_WIDTH];
    __shared__ int B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float pValue = 0;

    for (int m=0; m < width / TILE_WIDTH; m++)
    {
        A[ty][tx] = a[row * width + m * TILE_WIDTH + tx];
        B[ty][tx] = b[col + (m * TILE_WIDTH + ty) * width];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
            pValue += A[ty][k] * B[k][tx];
        __syncthreads();
    }

    c[row * width + col] = pValue;
}

__global__ void transpose_m(int* a, int* res, int width)
{
    int i = threadIdx.x * blockDim.x * blockIdx.x;
    int j = threadIdx.y * blockDim.y * blockIdx.y;

    res[j * width + i] = a[i * width + j];
}

#pragma endregion

class CudaMatrix
{
#pragma Fields, constructor and basic methods
    int* data;
    int width, height;
private:
    /// <summary>
    /// Generic method used to execute various cuda kernels
    /// </summary>
    /// <param name="b_matrix">Second operand</param>
    /// <param name="kernel">kernel to execute. Must have strict params (int*, int*, int*, int matrix_width)</param>
    /// <param name="blockDim">Block dimension. Default is one</param>
    /// <param name="threadDim">Thread dimension. Default is width, height.</param>
    /// <returns></returns>
    CudaMatrix ExecuteCudaKernel(
        CudaMatrix b_matrix,
        void (*kernel)(int* a, int* b, int* c, int width),
        dim3 threadDim = 0,
        dim3 blockDim = 1)
    {
        if (threadDim.x == 0)
            threadDim = dim3(this->height, this->width);

        CudaMatrix res(this->width, this->height);

        int* dev_a, * dev_b, * dev_c;

        int array_size = this->width * this->height * sizeof(int);

        cudaMalloc((void**)&dev_a, array_size);
        cudaMalloc((void**)&dev_b, array_size);
        cudaMalloc((void**)&dev_c, array_size);

        cudaMemcpy(dev_a, this->data, array_size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b_matrix.data, array_size, cudaMemcpyHostToDevice);


        kernel<< <blockDim, threadDim >> > (dev_a, dev_b, dev_c, this->width);
        cudaThreadSynchronize();

        cudaMemcpy(res.data, dev_c, array_size, cudaMemcpyDeviceToHost);

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);

        return res;
    }

public:
    CudaMatrix(int w, int h = 0)
    {
        if (h == 0) h = w;

        this->width = w;
        this->height = h;
        this->data = new int[h*w];
    }

    void PopulateMatrix(int precision = 4)
    {
        this->data = StructHelperMethods::GenerateArray(this->width * this->height, precision);
    }

    void Print()
    {
        cout << StructHelperMethods::PrintArrayAsMatrix(this->data, this->width, this->height);
    }

    CudaMatrix operator+(const CudaMatrix& operand)
    {
        if (this->width != operand.width || this->height != operand.height)
            throw "Matrix must have same dimensions!";

        return ExecuteCudaKernel(operand, add_m);
    }

    CudaMatrix operator-(const CudaMatrix& operand)
    {
        if (this->width != operand.width || this->height != operand.height)
            throw "Matrix must have same dimensions!";

        return ExecuteCudaKernel(operand, sub_m);
    }

    CudaMatrix operator*(const CudaMatrix& operand)
    {
        if (this->width != operand.width || this->height != operand.height)
            throw "Matrix must have same dimensions!";

        if (USE_SHARED_MEMORY)
        {
            return ExecuteCudaKernel(
                operand, mul_m_shared,
                dim3(TILE_WIDTH, TILE_WIDTH),
                dim3(this->width / TILE_WIDTH, this->width / TILE_WIDTH));
        }
        else if (MUL_USING_TILES)
        {
            return ExecuteCudaKernel(
                operand, mul_m_tiles,
                dim3(TILE_WIDTH, TILE_WIDTH),
                dim3(this->width / TILE_WIDTH, this->width / TILE_WIDTH));
        }

        return ExecuteCudaKernel(operand, mul_m);
    }

    /// <summary>
    /// Transpose
    /// </summary>
    /// <param name="add"></param>
    /// <returns></returns>
    void Transpose()
    {
        if (this->width != this->height)
            throw "This matrix cant be transposed!";

        int* dev_a, * dev_b;

        int array_size = this->width * this->height * sizeof(int);

        cudaMalloc((void**)&dev_a, array_size);
        cudaMalloc((void**)&dev_b, array_size);

        cudaMemcpy(dev_a, this->data, array_size, cudaMemcpyHostToDevice);


        transpose_m << <1, dim3(this->width, this->height) >> > (dev_a, dev_b, this->width);

        cudaThreadSynchronize();

        cudaMemcpy(this->data, dev_b, array_size, cudaMemcpyDeviceToHost);
    }
};