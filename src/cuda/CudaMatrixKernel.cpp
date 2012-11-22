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

void CudaMatrixKernel::startup(const std::vector<std::string>& arguments)
{
    if (arguments.size() != 2)
    {
        throw runtime_error("CUDA matrix multiplication needs 3 matrix files as arguments!");
    }

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

    cublas->setMatrix(matrixARows, matrixACols, sizeof(float) * matrixASize, matrixA.buffer(), matrixARows, matrixMemA, matrixARows);
    cublas->setMatrix(matrixACols, matrixBCols, sizeof(float) * matrixBSize, matrixB.buffer(), matrixACols, matrixMemB, matrixACols);

}

void CudaMatrixKernel::run()
{
    cublas->Sgemm(
        CUBLAS_OP_T, CUBLAS_OP_T,
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
    size_t outputSize = rows * cols;
    Matrix<float> targetMatrix(rows, cols);
    cublas->getMatrix(rows, cols, sizeof(float) * outputSize, outputMatrix, rows, targetMatrix.buffer(), rows);

    MatrixHelper::writeMatrixTo(outputFilename, targetMatrix);
}

std::shared_ptr<BenchmarkKernel> createKernel()
{
    BenchmarkKernel* kernel = new CudaMatrixKernel();
    return std::shared_ptr<BenchmarkKernel>(kernel);
}
