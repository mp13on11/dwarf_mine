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

    matrixC = Matrix<float>(matrixARows, matrixBCols);
}

void SMPMatrixKernel::run()
{
    cblas_sgemm(
        CblasRowMajor, 
        CblasNoTrans, 
        CblasNoTrans, 
        matrixARows, 
        matrixBCols, 
        matrixACols, 
        ALPHA, 
        matrixA.buffer(), 
        matrixACols, 
        matrixB.buffer(), 
        matrixBCols, 
        BETA, 
        matrixC.buffer(),
        matrixBCols);
}

void SMPMatrixKernel::shutdown(const std::string& outputFilename)
{
    MatrixHelper::writeMatrixTo(outputFilename, matrixC);
}

std::shared_ptr<BenchmarkKernel> createKernel()
{
    return std::shared_ptr<BenchmarkKernel>(new SMPMatrixKernel());
}
