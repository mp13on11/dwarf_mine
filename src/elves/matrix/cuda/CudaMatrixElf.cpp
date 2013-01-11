#include "CudaMatrixElf.h"
#include <cuda-utils/Memory.h>
#include "MatrixMultiplication.h"
#include <iostream>
#include <vector>
#include <cmath>
#include "../MatrixHelper.h"
#include "../Matrix.h"

MatrixElf::MatrixT CudaMatrixElf::multiply(const MatrixT& left, const MatrixT& right)
{
    using namespace std;

    int leftRows = left.rows();
    int rightCols = right.columns();
    int middle = left.columns();

    size_t leftSize = leftRows * middle;
    size_t rightSize = middle * rightCols;
    size_t resultSize = leftRows * rightCols;
    vector<float> result_h(resultSize);

    CudaUtils::Memory<float> left_d(leftSize);
    CudaUtils::Memory<float> right_d(rightSize);
    CudaUtils::Memory<float> result_d(resultSize);

    left_d.transferFrom(left.buffer());
    right_d.transferFrom(right.buffer());
    result_d.transferFrom(result_h.data());

    gemm(leftRows, rightCols, middle, left_d.get(), right_d.get(), result_d.get());

    result_d.transferTo(result_h.data());

    MatrixT resultMatrix(leftRows, rightCols, std::move(result_h));
    return resultMatrix;
}
