#include "mpi/MpiMatrixMultiplicationBenchmarkKernel.h"
#include "MismatchedMatricesException.h"

#include <mpi.h>
#include <sstream>

using namespace std;

const int MpiMatrixMultiplicationBenchmarkKernel::ROOT_RANK = 0;

MpiMatrixMultiplicationBenchmarkKernel::MpiMatrixMultiplicationBenchmarkKernel() :
        rank(MPI::COMM_WORLD.Get_rank())
{
}

void MpiMatrixMultiplicationBenchmarkKernel::startup(const vector<string>& arguments)
{
    if (rank != ROOT_RANK)
        return;

    left = readMatrixFrom(arguments[0]);
    right = readMatrixFrom(arguments[0]);

    if (left.columns() != right.rows())
        throw MismatchedMatricesException(left.columns(), right.rows());

    result = Matrix<float>(left.rows(), right.columns());
}

void MpiMatrixMultiplicationBenchmarkKernel::run()
{
    broadcastSizes();
    scatterMatrices();
    multiply();
    gatherResult();
}

void MpiMatrixMultiplicationBenchmarkKernel::shutdown(const string& outputFileName)
{
    if (rank != ROOT_RANK)
        return;

    writeMatrixTo(outputFileName, result);
}

void MpiMatrixMultiplicationBenchmarkKernel::broadcastSizes()
{
}

void MpiMatrixMultiplicationBenchmarkKernel::scatterMatrices()
{
}

void MpiMatrixMultiplicationBenchmarkKernel::multiply()
{
}

void MpiMatrixMultiplicationBenchmarkKernel::gatherResult()
{
}
