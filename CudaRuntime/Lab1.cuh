#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "HelpMethods.h"

using namespace std;


#pragma region Device and Global methods

__global__ void prvi(int* a, int* b, int x, int a_size, int b_size, int* res)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < a_size)
    {
        res[index] = a[index] * x;
    }
    if(index < b_size)
    {
        res[index] = b[index] + res[index];
    }
}

__global__ void drugi(int** a, int** b, int** res, int k, int p)
{
    int val = a[threadIdx.x][threadIdx.y] + b[threadIdx.x][threadIdx.y];
    if (val > k) val = p;
    res[threadIdx.x][threadIdx.y] = val;
}

__global__ void drugi(int* a, int* b, int* c, int k, int p)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    int val = a[index] + b[index];
    if (val > k) val = p;
    c[index] = val;
}

__global__ void drugi(int* a, int* b, int* c, int dim, int k, int p)
{
    int index = threadIdx.x * dim + threadIdx.y;

    int val = a[index] + b[index];
    if (val > k) val = p;
    c[index] = val;
}

#pragma endregion

class Lab1
{
#pragma region Common

private:
    void Compare(int a[], int b[])
    {
        int a_size = sizeof(a) / sizeof(int);
        int b_size = sizeof(b) / sizeof(int);

        if (a_size != b_size)
        {
            cout << "Cak se i velicine razlikuju!" << endl;
            return;
        }

        bool same = true;
        int index = 0;
        while (same && index < a_size)
        {
            same = a[index] == b[index];
            index++;
        }

        cout << (same ? "" : "Ne ") << "Slazu se!" << endl;
    }

#pragma endregion

public:
    Lab1()
    {
        cout << "================ LAB 1 CUDA ====================" << endl;
    }
public:
    /// <summary>
    /// Napisati CUDA program koji računa sledeći izraz: A*x + B, gde su A i B vektori, a x skalar. Napisati
    /// kod za testiranje validnosti rezultata, upoređivanjem sa vrednostima dobijenim izvršavanjem
    /// sekvencijalnog koda koji izračunava isti izraz.Pripremiti se za diskusiju ponašanja programa u
    /// zavisnosti od broja blokova i broja niti u okviru jednog bloka.
    /// </summary>
    void Prvi()
    {
        int a_size, b_size;
        cout << "Unesi velicine nizova:" << endl;
        cout << "A: ";
        cin >> a_size;
        cout << "B: ";
        cin >> b_size;
        cout << endl;

        int x = StructHelperMethods::GenerateNumber();
        int* a = StructHelperMethods::GenerateArray(a_size);
        int* b = StructHelperMethods::GenerateArray(b_size);

        cout << "X: " << x << ", A: " << StructHelperMethods::PrintArray(a, a_size) << ", B: " << StructHelperMethods::PrintArray(b, b_size) << endl;

        // CPU
        int res_size = a_size > b_size ? a_size : b_size;
        int* res = new int[10];
        for (int i= 0; i< res_size;i++)
        {
            if (i < a_size)
                res[i] = a[i] * x;
            if (i < b_size)
                res[i] += b[i];
        }

        cout << "CPU: (A*x + B): " << StructHelperMethods::PrintArray(res, res_size) << endl;

        // GPU
        int* dev_a;
        int* dev_b;
        int* dev_res, *h_res = new int[res_size];

        cudaMalloc((void**)&dev_a, a_size * sizeof(int));
        cudaMalloc((void**)&dev_b, b_size * sizeof(int));
        cudaMalloc((void**)&dev_res, res_size * sizeof(int));

        cudaMemcpy(dev_a, a, a_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, b_size * sizeof(int), cudaMemcpyHostToDevice);

        prvi <<<1, res_size >>> (dev_a, dev_b, x, a_size, b_size, dev_res);

        cudaMemcpy(h_res, dev_res, res_size * sizeof(int), cudaMemcpyDeviceToHost);
        cout << "GPU: (A*x + B): " << StructHelperMethods::PrintArray(h_res, res_size) << endl;

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_res);

        Compare(res, h_res);
    }

    /// <summary>
    /// Napisati CUDA program koji računa sledeći izraz: A + B, gde su A i B kvadratne matrice. U
    /// novodobijenoj matrici sve vrednosti koje su veće od nekog k zameniti brojem p.Brojeve k i p
    /// unosi korisnik.Napisati kod za testiranje validnosti rezultata, upoređivanjem sa vrednostima
    /// dobijenim izvršavanjem sekvencijalnog koda koji izračunava isti izraz.Pripremiti se za diskusiju
    /// ponašanja programa u zavisnosti od broja blokova i broja niti u okviru jednog bloka.
    /// </summary>
    void Drugi()
    {
        int size, k, p;
        cout << "Matrix size: "; cin >> size;
        cout << "K: "; cin >> k;
        cout << "P: "; cin >> p;

        int** a_matrix = new int* [size];
        int** b_matrix = new int* [size];
        for (int i = 0; i < size; i++)
        {
            b_matrix[i] = new int[size];
            a_matrix[i] = new int[size];
        }

        a_matrix = StructHelperMethods::GenerateMatrix(size, size, 4);
        b_matrix = StructHelperMethods::GenerateMatrix(size, size, 4);
        cout << "Generated matrix A: " << endl;
        cout << StructHelperMethods::PrintMatrix(a_matrix, size, size);
        cout << "Generated matrix B: " << endl;
        cout << StructHelperMethods::PrintMatrix(b_matrix, size, size);

        // CPU:
        int** res_matrix = new int*[size];
        for (int i = 0; i < size; i++)
        {
            res_matrix[i] = new int[size];
            for (int j = 0; j< size; j++)
            {
                int val = a_matrix[i][j] + b_matrix[i][j];
                if (val > k) val = p;
                res_matrix[i][j] = val;
            }
        }

        cout << "CPU: result matrix: " << endl;
        cout << StructHelperMethods::PrintMatrix(res_matrix, size, size);

        // GPU:

        #pragma region Fail
        // Ovde sam pokusao da posaljem vrednosti kao matrice, Dont be me! Moguce je ali svi proprucuju koriscenje iskljucivo nizova.

        /*   int** dev_a, **dev_b, **dev_res;
        int** h_res_matrix = new int*[size];
        for (int i = 0; i < size; i++)
            h_res_matrix[i] = new int[size];

        int arr_size = size * size * sizeof(int);


        CudaErrorHandling::HANDLE_ERROR( cudaMalloc((void***)&dev_a, size));
        CudaErrorHandling::HANDLE_ERROR( cudaMalloc((void***)&dev_b, size));
        CudaErrorHandling::HANDLE_ERROR( cudaMalloc((void***)&dev_res, size));

        for (int i = 0; i < size; i++) {
            cudaMalloc((void**)&(dev_a[i]), size * sizeof(int));
            cudaMalloc((void**)&(dev_b[i]), size * sizeof(int));
            cudaMalloc((void**)&(dev_res[i]), size * sizeof(int));
            cudaMemcpy(dev_a[i], a_matrix[i], size * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_b[i], b_matrix[i], size * sizeof(int), cudaMemcpyHostToDevice);
        }

        dim3 threads(size, size, 1);

        drugi << <1, threads>> > (dev_a, dev_b, dev_res, k, p);
        CudaErrorHandling::HANDLE_ERROR(cudaDeviceSynchronize());

        CudaErrorHandling::HANDLE_ERROR(cudaMemcpy(h_res_matrix, dev_res, arr_size, cudaMemcpyDeviceToHost));

        cout << "GPU: result matrix: " << endl;
        cout << StructHelperMethods::PrintMatrix(h_res_matrix, size, size);

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_res);*/

    #pragma endregion

        int* dev_a, * dev_b, * dev_c;
        int* h_c = new int[size*size];

        int arr_size = size * size * sizeof(int);
        cudaMalloc((void**)&dev_a, arr_size);
        cudaMalloc((void**)&dev_b, arr_size);
        cudaMalloc((void**)&dev_c, arr_size);

        cudaMemcpy(dev_a, StructHelperMethods::SquaredMatrixToArray(a_matrix, size), arr_size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, StructHelperMethods::SquaredMatrixToArray(b_matrix, size), arr_size, cudaMemcpyHostToDevice);

        dim3 threads(size, size, 1);

        drugi << <1, threads >> > (dev_a, dev_b, dev_c, size, k, p);
        cudaThreadSynchronize();

        cudaMemcpy(h_c, dev_c, arr_size, cudaMemcpyDeviceToHost);

        cout << "GPU: result matrix: " << endl;
        cout << StructHelperMethods::PrintArrayAsMatrix(h_c, size, size);
    }
};