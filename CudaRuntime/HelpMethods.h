#pragma once
#include <cstdlib>
#include <string>

static class StructHelperMethods
{
public:
	static int GenerateNumber(int range = 10)
	{
		return rand() % range;
	}

	static int* GenerateArray(int size, int range = 10)
	{
		int* arr = new int[size];

		for (int i = 0; i < size; i++)
			arr[i] = rand() % range;
		return arr;
	}

	static int** GenerateMatrix(int w, int h, int range = 10)
	{
		int** matrix = new int * [h];
		for (int i = 0; i < h; i++)
			matrix[i] = new int[w];

		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
				matrix[j][i] = rand() % range;

		return matrix;
	}

	static int* SquaredMatrixToArray(int **mat, int size)
	{
		int* arr = new int[size * size];
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
				arr[i*size + j] = mat[i][j];
		}

		return arr;
	}

	static int** ArrayToSquaredMatrix(int arr[], int size)
	{
		if (sizeof(arr) / sizeof(int) != size * size)
		{
			throw "Array cant be converted to matrix!";
		}

		int** res = new int* [size];
		for (int i = 0; i < size; i ++)
		{
			res[i] = new int[size];
			for (int j = 0; j < size; j++)
				res[i][j] = arr[i * j + j];
		}
		return res;
	}


	static string PrintArray(int* arr, int size)
	{
		string res = "[ ";

		for (int i = 0; i < size; i++)
		{
			res += std::to_string(arr[i]);
			if (i < size-1)
				res += ',';
			res += ' ';
		}
		res += "]\0";

		return res;
	}

	static string PrintMatrix(int** matrix, int w, int h)
	{
		string res;

		for (int i = 0; i < h; i++)
		{
			res += "| ";

			for (int j = 0; j < w; j++)
			{
				res += std::to_string(matrix[i][j]);

				if (j < w - 1)
					res += ',';
				res += ' ';
			}

			res += "|\n";
		}

		res += "\0";

		return res;
	}

	static string PrintArrayAsMatrix(int* arr, int w, int h)
	{
		string res;
		for (int i = 0; i < h; i++)
		{
			res += "| ";

			for (int j = 0; j < w; j++)
			{
				res += std::to_string(arr[i * w + j]);

				if (j < w - 1)
					res += ',';
				res += ' ';
			}

			res += "|\n";
		}

		res += "\0";

		return res;
	}
};

static class CudaErrorHandling
{
public:
	static void HANDLE_ERROR(cudaError_t call)
	{
		cudaError_t ret = call;
		switch (ret)
		{
		case cudaSuccess:
			break;
		case cudaErrorInvalidValue:
			printf("ERROR: InvalidValue: %i.\n", __LINE__);
			break;
		case cudaErrorInvalidMemcpyDirection:
			printf("ERROR: Invalid memcpy direction: %i.\n", __LINE__);
			break;
		default:
			printf("ERROR>line: %i.%d' ‘ %s\n", __LINE__, ret, cudaGetErrorString(ret));
			break;
		}
	}

	static void HANDLE_ERROR_EXIT(cudaError_t call)
	{
		cudaError_t ret = call;
		switch (ret)
		{
		case cudaSuccess:
			break;
		case cudaErrorInvalidValue:
			printf("ERROR: InvalidValue:%i.\n", __LINE__);
			exit(-1);
			break;
		case cudaErrorInvalidMemcpyDirection:
			printf("ERROR:Invalid memcpy direction:%i.\n", __LINE__);
			exit(-1);
			break;
		default:
			printf("ERROR>line:%i.%d' ‘ %s\n", __LINE__, ret, cudaGetErrorString(ret));
			exit(-1);
			break;
		}
	}

};
