#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#pragma region Device and Global Methods

__device__ int compare(float a, float b)
{
	return a == b ? 1 : 0;
}

/// <summary>
/// Ciklicno pretrazivanje niza
/// </summary>
/// <param name="soughtNumber"></param>
/// <param name="in_arr"></param>
/// <param name="out_arr"></param>
/// <param name="numberOfThreadsPerBlock"></param>
/// <returns></returns>
__global__ void compute(float soughtNumber, float* in_arr, float* out_arr, int numberOfThreadsPerBlock)
{
	out_arr[threadIdx.x] = 0;

	for (int i=0; i < numberOfThreadsPerBlock; i++)
	{
		float val = in_arr[i * blockDim.x + threadIdx.x];
		out_arr[threadIdx.x] += compare(soughtNumber, val);
	}
}

__global__ void add(float* a, float* b, float* c)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	c[index] = a[index] + b[index];
}

__global__ void mul(float* a, float* b, float* c)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	c[index] = a[index] * b[index];
}

#pragma endregion

class CudaVector
{
#pragma region Fields, Constructor and Basic methods
private:
	int h_size;
	float* h_vector;
public:
	CudaVector(int dim)
	{
		this->h_size = dim;
		this->h_vector = new float[dim];
	}
	void SetAtIndex(float val, int index)
	{
		this->h_vector[index] = val;
	}
	float DeleteFromIndex(int index)
	{
		float tmp = this->h_vector[index];
		this->h_vector[index] = 0;
		return tmp;
	}
	void SetValues(float values[], int dim)
	{
		this->h_vector = values;
		this->h_size = dim;
	}

	void Print()
	{
		cout << "[ ";
		for (int i = 0; i < this->h_size; i++)
			if (i < h_size -1)
				cout << h_vector[i] << ", ";
			else cout << h_vector[i];
		cout << " ]" << endl;
	}
#pragma endregion

#pragma region Math methods
public:
	__host__ int FindNumberOfOccurrences(float el)
	{
		int blockSize = 3;
		int numberOfThreads = this->h_size / blockSize;
		float *dev_vector;
		float* dev_out_sum, *host_out_sum = new float[blockSize];

		cudaMalloc((void**)&dev_vector, this->h_size * sizeof(float));
		cudaMalloc((void**)&dev_out_sum, blockSize * sizeof(float));
		cudaMemcpy(dev_vector, this->h_vector, this->h_size * sizeof(float), cudaMemcpyHostToDevice);

		compute <<<1, blockSize >>> (el, dev_vector, dev_out_sum, numberOfThreads);
		cudaThreadSynchronize();

		cudaMemcpy(host_out_sum, dev_out_sum, blockSize * sizeof(float), cudaMemcpyDeviceToHost);

		int count = 0;
		for (int i = 0; i < blockSize; i++)
		{
			count += host_out_sum[i];
		}
		return count;
	}

	__host__ void Add(CudaVector* vector1, CudaVector* vector2)
	{
		float* dev_vector1;
		float* dev_vector2;
		float* dev_res_vector;

		int larger = vector1->h_size > vector2->h_size ? vector1->h_size : vector2->h_size;
		this->h_vector = new float[larger];
		this->h_size = larger;

		cudaMalloc((void**)&dev_vector1, vector1->h_size * sizeof(float));
		cudaMalloc((void**)&dev_vector2, vector2->h_size * sizeof(float));
		cudaMalloc((void**)&dev_res_vector, larger * sizeof(float));

		cudaMemcpy(dev_vector1, vector1->h_vector, vector1->h_size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_vector2, vector2->h_vector, vector2->h_size * sizeof(float), cudaMemcpyHostToDevice);

		add<<<1,larger>>>(dev_vector1, dev_vector2, dev_res_vector);

		cudaMemcpy(this->h_vector, dev_res_vector, larger * sizeof(float), cudaMemcpyDeviceToHost);
	}

	__host__ void Mul(CudaVector* vector1, CudaVector* vector2)
	{
		float* dev_vector1;
		float* dev_vector2;
		float* dev_res_vector;

		int larger = vector1->h_size > vector2->h_size ? vector1->h_size : vector2->h_size;
		this->h_vector = new float[larger];
		this->h_size = larger;

		cudaMalloc((void**)&dev_vector1, vector1->h_size * sizeof(float));
		cudaMalloc((void**)&dev_vector2, vector2->h_size * sizeof(float));
		cudaMalloc((void**)&dev_res_vector, larger * sizeof(float));

		cudaMemcpy(dev_vector1, vector1->h_vector, vector1->h_size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_vector2, vector2->h_vector, vector2->h_size * sizeof(float), cudaMemcpyHostToDevice);

		mul << <1, larger >> > (dev_vector1, dev_vector2, dev_res_vector);

		cudaMemcpy(this->h_vector, dev_res_vector, larger * sizeof(float), cudaMemcpyDeviceToHost);
	}
#pragma endregion
};