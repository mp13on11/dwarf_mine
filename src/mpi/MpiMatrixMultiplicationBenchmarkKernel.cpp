#include "mpi/MpiMatrixMultiplicationBenchmarkKernel.h"
#include "lib/tools/MismatchedMatricesException.h"
#include "lib/tools/MatrixHelper.h"

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

    left = MatrixHelper::readMatrixFrom(arguments[0]);
    right = MatrixHelper::readMatrixFrom(arguments[0]);

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

    MatrixHelper::writeMatrixTo(outputFileName, result);
}

void MpiMatrixMultiplicationBenchmarkKernel::broadcastSizes()
{
    size_t sizes[4] = {
            left.rows(), left.columns(), right.rows(), right.columns()
        };
    MPI::COMM_WORLD.Bcast(sizes, 4 * sizeof(size_t), MPI::CHAR, ROOT_RANK);

    if (rank != ROOT_RANK)
    {
        left = Matrix<float>(sizes[0], sizes[1]);
        right = Matrix<float>(sizes[2], sizes[3]);
        result = Matrix<float>(sizes[0], sizes[3]);
    }
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
