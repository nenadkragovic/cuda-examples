#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#pragma region Device and Global Methods


#pragma endregion

template <class T> class Matrix
{
#pragma Fields, constructor and basic methods
	T** matrix;
	int width, height;
public:
	Matrix(int w, int h)
	{
		this->width = w;
		this->height = h;
		this->matrix = new T*[h];
		for (int i = 0; i < h; i++)
			this->matrix[i] = new T[w];
	}

	void PopulateMatrix()
	{
		for (int i = 0; i < this->height; i++)
			for (int j = 0; j < this->width; j++)
				this->matrix[j][i] = rand() % 10;
	}

	void Print()
	{
		for (int i = 0; i < this->height; i++)
		{
			cout << "| ";
			for (int j = 0; j < this->width; j++)
			{
				cout << this->matrix[j][i];
				if (j < this->width - 1)
					cout << ", ";
			}
			cout << " |" << endl;
		}
	}

};