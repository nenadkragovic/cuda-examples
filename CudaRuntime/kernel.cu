#include <iostream>
#include "CudaVector.cuh"
using namespace std;

int main()
{
	CudaVector* v1 = new CudaVector(10);
	v1->SetValues(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 10);
	v1->Print();
	CudaVector* v2 = new CudaVector(10);
	v2->SetValues(new float[] { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}, 10);
	v2->Print();
	CudaVector* v3 = new CudaVector(0);
	v3->Add(v1, v2);
	v3->Print();


	CudaVector* v4 = new CudaVector(10);
	v4->SetValues(new float[] { 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 2, 4}, 12);
	v4->Print();
	cout << "Number od 2 in array: " << v4->FindNumberOfOccurrences(2) << endl;

	return 0;
}
