#include <smp/SMPMatrixKernel.h>
#include <iostream>
#include <tools/MismatchedMatricesException.h>
#include <tools/MatrixHelper.h>

extern "C" {
#include "cblas.h"
}

using namespace::std;

const float ALPHA = 1.0f;
const float BETA = 0.0f;

size_t SMPMatrixKernel::requiredInputs() const
{
    return 2;
}

void SMPMatrixKernel::startup(const std::vector<std::string>& arguments)
{
    matrixA = MatrixHelper::readMatrixFrom(arguments[0]);
    matrixB = MatrixHelper::readMatrixFrom(arguments[1]);

    matrixARows = matrixA.rows();
    matrixACols = matrixA.columns();
    matrixBRows = matrixB.rows();
    matrixBCols = matrixB.columns();

    if (matrixACols != matrixBRows)
        throw MismatchedMatricesException(matrixACols, matrixBRows);

}

void SMPMatrixKernel::run()
{
    const float* matA = matrixA.buffer();
    const float* matB = matrixA.buffer();
    matC = new float[matrixBRows * matrixACols];
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, matrixARows, matrixBCols, matrixACols, ALPHA, matA, matrixARows, matB, matrixBRows, BETA, matC, matrixACols);
}

void SMPMatrixKernel::shutdown(const std::string& outputFilename)
{
    size_t rows = matrixBRows;
    size_t cols = matrixACols;

    vector<float> data(&matC[0], &matC[rows*cols-1]);
    Matrix<float> targetMatrix(rows, cols, move(data));
    MatrixHelper::writeMatrixTo(outputFilename, targetMatrix);

    delete [] matC;
}

std::shared_ptr<BenchmarkKernel> createKernel()
{
    BenchmarkKernel* kernel = new SMPMatrixKernel();
    return std::shared_ptr<BenchmarkKernel>(kernel);
}
