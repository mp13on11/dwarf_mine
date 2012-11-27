#include <SMPMatrixKernel>
#include <iostream>
#include <tools/Matrix.h>
#include <tools/MismatchedMatricesException.h>
#include <tools/MatrixHelper.h>

extern "C" {
#include "cblas.h"
}

using namespace::std;

const float ALPHA = 1.0f;
const float BETA = 0.0f;

void SMPMatrixKernel::startup(const std::vector<std::string>& arguments)
{
    Matrix<float> matrixA = MatrixHelper::readMatrixFrom(arguments[0]);
    Matrix<float> matrixB = MatrixHelper::readMatrixFrom(arguments[1]);

    matrixARows = matrixA.rows();
    matrixACols = matrixA.columns();
    matrixBRows = matrixB.rows();
    matrixBCols = matrixB.columns();

    if (matrixACols != matrixBRows)
        throw MismatchedMatricesException(matrixACols, matrixBRows);

}

void SMPMatrixKernel::run()
{
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, matrixARows, matrixBCols, matrixACols, ALPHA, matrixA, matrixARows, matrixB, matrixBRows, BETA, matrixC, matrixACols);
}

void SMPMatrixKernel::shutdown(const std::string& outputFilename)
{
    size_t rows = matrixBRows;
    size_t cols = matrixACols;

    Matrix<float> targetMatrix(rows, cols, move(outputBuffer));
    MatrixHelper::writeMatrixTo(outputFilename, targetMatrix);
}

std::shared_ptr<BenchmarkKernel> createKernel()
{
    BenchmarkKernel* kernel = new SMPMatrixKernel();
    return std::shared_ptr<BenchmarkKernel>(kernel);
}
