#include "CudaMatrixElf.h"
#include "Memory.h"
#include "MatrixMultiplication.h"
#include <iostream>
#include <vector>
#include <cmath>
#include "../MatrixHelper.h"
#include "../Matrix.h"

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
            if (fabs(out[r*n+c]-sum) > 0.1)
            {
                std::cout << "fehler: (" << r << ", " << c << ") " << sum << " " << out[r*n+c] << std::endl;
                return;
            }
        }
    }
}

void CudaMatrixElf::run(std::istream& input, std::ostream& output)
{
    using namespace std;

    //Matrix<float> leftMatrix = MatrixHelper::readMatrixFrom(input);
    //Matrix<float> rightMatrix = MatrixHelper::readMatrixFrom(input);

    Matrix<float> leftMatrix(3, 2, vector<float>{1, 2, 3, 4, 5, 6});
    Matrix<float> rightMatrix(2, 3, vector<float>{1, 2, 3, 4, 5, 6});
   
    leftMatrix.addPadding(32);
    rightMatrix.addPadding(32);

    MatrixHelper::validateMultiplicationPossible(leftMatrix, rightMatrix);

    int leftRows = leftMatrix.rows();
    int rightCols = rightMatrix.columns();
    int middle = leftMatrix.columns();

    size_t leftSize = leftRows * middle;
    size_t rightSize = middle * rightCols;
    size_t resultSize = leftRows * rightCols;
    vector<float> result_h(resultSize);
    for (int i = 0; i < resultSize; ++i)
    {
        result_h.at(i) = 0;
    }

    CudaUtils::Memory<float> left_d(leftSize);
    CudaUtils::Memory<float> right_d(rightSize);
    CudaUtils::Memory<float> result_d(resultSize);

    left_d.transferFrom(leftMatrix.buffer());
    right_d.transferFrom(rightMatrix.buffer());
    result_d.transferFrom(result_h.data());

    for (int i=0; i < 1; ++i)
        gemm(leftRows, rightCols, middle, left_d.get(), right_d.get(), result_d.get(), 32);

    result_d.transferTo(result_h.data());

    mul(leftRows, rightCols, middle, leftMatrix.buffer(), rightMatrix.buffer(), result_h.data());
    
    Matrix<float> resultMatrix(leftRows, rightCols, std::move(result_h));
    MatrixHelper::writeMatrixTo(output, resultMatrix);
}
