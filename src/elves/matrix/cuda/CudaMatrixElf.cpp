#include "CudaMatrixElf.h"
#include "Memory.h"
#include "MatrixMultiplication.h"
#include <iostream>
#include <vector>
#include <cmath>

void mul(int m, int n, int k, float* left, float* right, float* out)
{
    for (int r = 0; r < m; ++r)
    {
        for (int c = 0; c < n; ++c)
        {
            float sum = 0;
            for (int i=0; i < k; ++i)
            {
                sum += left[r*k+i] * right[n*i+c];
            }
            if (fabs(out[r*n+c]-sum) > 0.1) {
                std::cout << "alles scheisse: (" << r << ", " << c << ") " << sum << " " << out[r*n+c] << std::endl;
                return;
            }
        }
    }
}

void CudaMatrixElf::test()
{
	using namespace std;

	int leftRows = 500;
	int rightCols = 800;
	int middle = 400;

	vector<float> left(leftRows*middle);
	vector<float> right(middle*rightCols);
	vector<float> result_h(leftRows*rightCols);

	srand( time(NULL) );
	for (int i=0; i < leftRows*middle; ++i)
		left[i] = (float) rand() / RAND_MAX;

	for (int i=0; i < middle*rightCols; ++i)
		right[i] = (float) rand() / RAND_MAX;

	CudaUtils::Memory<float> left_d(leftRows*middle);
	CudaUtils::Memory<float> right_d(middle*rightCols);
	CudaUtils::Memory<float> result_d(leftRows*rightCols);

	left_d.transferFrom(left.data());
	right_d.transferFrom(right.data());

	for (int i=0; i < 1; ++i)
		gemm(leftRows, rightCols, middle, left_d.get(), right_d.get(), result_d.get());

	result_d.transferTo(result_h.data());

	mul(leftRows, rightCols, middle, left.data(), right.data(), result_h.data());

	cout << result_h[0] << " " << result_h[1] << endl;
	cout << result_h[2] << " " << result_h[3] << endl;

	[](){cout << "krasser shit!" << endl;}();
}
