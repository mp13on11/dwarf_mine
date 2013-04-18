/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

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
