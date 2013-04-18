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

#pragma once

#include "GoldMatrixElf.h"
#include "matrix/Matrix.h"
#include "matrix/MatrixElf.h"
#include "matrix/MatrixHelper.h"

#include <cstdlib>
#include <functional>
#include <gtest/gtest.h>
#include <memory>
#include <random>

template<typename T>
class MatrixMultiplyTest : public testing::Test
{
protected:
    //
    // Members
    virtual void SetUp();
    virtual void TearDown();

    Matrix<float> createRandomMatrix(size_t rows, size_t columns);
    void initRandom(uint seed);
    Matrix<float> executeMultiplication(MatrixElf& elf, const Matrix<float>& a, const Matrix<float>& b);

    //
    // Instance variables
    std::function<float()> generator;

    std::string inputAFile;
    std::string inputBFile;
    std::string outputFile;

    std::unique_ptr<MatrixElf> referenceImplementation;
    std::unique_ptr<MatrixElf> currentImplementation;
};

template<typename T>
void MatrixMultiplyTest<T>::SetUp()
{
    inputBFile = "b.txt";
    outputFile = "c.txt";
    currentImplementation.reset(new T());
    referenceImplementation.reset(new GoldMatrixElf());
}

template<typename T>
void MatrixMultiplyTest<T>::TearDown()
{
    remove(inputAFile.c_str());
    remove(inputBFile.c_str());
    remove(outputFile.c_str());
}

template<typename T>
Matrix<float> MatrixMultiplyTest<T>::createRandomMatrix(size_t rows, size_t columns)
{
    Matrix<float> m(rows, columns);
    MatrixHelper::fill(m, generator);
    return m;
}

template<typename T>
void MatrixMultiplyTest<T>::initRandom(uint seed)
{
    auto distribution = std::uniform_real_distribution<float>(-100, +100);
    auto engine = std::mt19937(seed);
    generator = bind(distribution, engine);
}

template<typename T>
Matrix<float> MatrixMultiplyTest<T>::executeMultiplication(MatrixElf& elf, const Matrix<float>& a, const Matrix<float>& b)
{
    return elf.multiply(a, b);
}

#include "matrix/smp/SMPMatrixElf.h"

#ifdef HAVE_CUDA
#include "matrix/cuda/CudaMatrixElf.h"
typedef testing::Types<CudaMatrixElf, SMPMatrixElf> MatrixElfTypes;
#else
typedef testing::Types<SMPMatrixElf> MatrixElfTypes;
#endif

TYPED_TEST_CASE(MatrixMultiplyTest, MatrixElfTypes);
