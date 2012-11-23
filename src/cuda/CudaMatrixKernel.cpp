#include "CudaMatrixKernel.h"
#include "Cublas.h"
#include <iostream>
#include <stdexcept>
#include <tools/Matrix.h>
#include <tools/MismatchedMatricesException.h>
#include <tools/MatrixHelper.h>

using namespace std;
using namespace CudaUtils;

const float ALPHA = 1.0f;
const float BETA = 0.0f;

size_t CudaMatrixKernel::requiredInputs() const
{
    return 2;
}

void CudaMatrixKernel::startup(const std::vector<std::string>& arguments)
{
    Matrix<float> matrixA = MatrixHelper::readMatrixFrom(arguments[0]);
    Matrix<float> matrixB = MatrixHelper::readMatrixFrom(arguments[1]);

    matrixARows = matrixA.rows();
    matrixACols = matrixA.columns();
    size_t matrixBRows = matrixB.rows();
    matrixBCols = matrixB.columns();

    if (matrixACols != matrixBRows)
        throw MismatchedMatricesException(matrixACols, matrixBRows);

    size_t matrixASize = matrixARows * matrixACols;
    size_t matrixBSize = matrixACols * matrixBCols;
    size_t outputSize = matrixARows * matrixBCols;

    matrixMemA.reallocate(matrixASize);
    matrixMemB.reallocate(matrixBSize);
    outputMatrix.reallocate(outputSize);

    cublas.reset(new Cublas());

    cublas->setMatrix(matrixARows, matrixACols, sizeof(float), matrixA.buffer(), matrixARows, matrixMemA, matrixARows);
    cublas->setMatrix(matrixACols, matrixBCols, sizeof(float), matrixB.buffer(), matrixACols, matrixMemB, matrixACols);
}

void CudaMatrixKernel::run()
{
    cublas->Sgemm(
        CUBLAS_OP_N, CUBLAS_OP_N,
        matrixARows, matrixBCols, matrixACols,
        &ALPHA,
        matrixMemA, matrixARows,
        matrixMemB, matrixACols,
        &BETA,
        outputMatrix, matrixARows
    );
}

void CudaMatrixKernel::shutdown(const std::string& outputFilename)
{
    size_t rows = matrixARows;
    size_t cols = matrixBCols;

    vector<float> outputBuffer(rows * cols);
    cublas->getMatrix(rows, cols, sizeof(float), outputMatrix, rows, outputBuffer.data(), rows);

    Matrix<float> targetMatrix(rows, cols, move(outputBuffer));
    MatrixHelper::writeMatrixTo(outputFilename, targetMatrix);
}

std::shared_ptr<BenchmarkKernel> createKernel()
{
    BenchmarkKernel* kernel = new CudaMatrixKernel();
    return std::shared_ptr<BenchmarkKernel>(kernel);
}
