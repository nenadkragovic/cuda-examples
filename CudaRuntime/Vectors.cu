#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;

template <class T> class Vector
{
private:
	int h_dim;
	T* h_vector;
public:
	Vector(int dim)
	{
		this->h_dim = dim;
		this->h_vector = new T[dim];
	}
	void SetAtIndex(T val, int index)
	{
		this->h_vector[index] = val;
	}
	T DeleteFromIndex(int index)
	{
		T tmp = this->h_vector[index];
		this->h_vector[index] = 0;
		return tmp;
	}

public:
	Vector* Add(Vector& vector_to_add)
	{

	}
};