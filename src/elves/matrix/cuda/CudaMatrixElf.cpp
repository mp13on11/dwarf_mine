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

void CudaMatrixElf::run(std::istream& input, std::ostream& output)
{
	using namespace std;

	int leftRows = 1060;
	int rightCols = 1060;
	int middle = 1060;

	size_t leftSize = leftRows * middle;
	size_t rightSize = middle * rightCols;
	size_t resultSize = leftRows * rightCols;

	vector<float> left(leftSize);
	vector<float> right(rightSize);
	vector<float> result_h(resultSize);

	srand( time(NULL) );
	for (int i=0; i < leftSize; ++i)
		left[i] = (float) rand() / RAND_MAX;

	for (int i=0; i < rightSize; ++i)
		right[i] = (float) rand() / RAND_MAX;

	for (int i=0; i < resultSize; ++i)
		result_h[i] = 0.0f;

	CudaUtils::Memory<float> left_d(leftSize);
	CudaUtils::Memory<float> right_d(rightSize);
	CudaUtils::Memory<float> result_d(resultSize);

	left_d.transferFrom(left.data());
	right_d.transferFrom(right.data());
	result_d.transferFrom(result_h.data());

	for (int i=0; i < 1; ++i)
		gemm(leftRows, rightCols, middle, left_d.get(), right_d.get(), result_d.get(), 32);

	result_d.transferTo(result_h.data());

	mul(leftRows, rightCols, middle, left.data(), right.data(), result_h.data());

	cout << result_h[0] << " " << result_h[1] << endl;
	cout << result_h[2] << " " << result_h[3] << endl;

	[](){cout << "krasser shit!" << endl;}();
}
