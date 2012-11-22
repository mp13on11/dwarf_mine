#include <iomanip>
#include <iostream>
#include "Matrix.h"

using namespace std;

int main()
{
	Matrix<int> matrix(5, 5);

	for (int i=0; i<5; i++)
	{
		for (int j=0; j<5; j++)
		{
			matrix(i, j) = i * 5 + j;
			cout << " " << setw(3) << matrix(i, j);
		}
		cout << endl;
	}
}
