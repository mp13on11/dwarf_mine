#include "CudaMatrixElf.h"
#include "Memory.h"
#include "MatrixMultiplication.h"
#include <iostream>
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
                std::cout << "alles scheisse" << sum << " " << out[r*n+c] << std::endl;
                return;
            }
        }
    }
}

void CudaMatrixElf::test()
{
	int m = 5;
	int n = 5;
	int k = 5;

	float* a = new float[m*k];
	float* b = new float[k*n];
	float* c_h = new float[m*n];

	srand( time(NULL) );
	for (int i=0; i < m*k; ++i)
	{
		a[i] = (float) rand() /RAND_MAX;
		b[i] = (float) rand() /RAND_MAX;
	}

	CudaUtils::Memory<float> a_d(m*k);
	CudaUtils::Memory<float> b_d(k*n);
	CudaUtils::Memory<float> c_d(m*n);

	a_d.transferFrom(a);
	b_d.transferFrom(b);

	for (int i=0; i < 1; ++i)
		gemm(m, n, k, a_d.get(), b_d.get(), c_d.get());

	c_d.transferTo(c_h);

	mul(m, n, k, a, b, c_h);

	std::cout << c_h[0] << " " << c_h[1] << std::endl;
	std::cout << c_h[2] << " " << c_h[3] << std::endl;

	[](){std::cout << "krasser shit!" << std::endl;}();
}
