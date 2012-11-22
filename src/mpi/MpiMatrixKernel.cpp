#include "mpi/MpiMatrixKernel.h"
#include "tools/MismatchedMatricesException.h"
#include "tools/MatrixHelper.h"

#include <mpi.h>
#include <sstream>

using namespace std;

const int MpiMatrixKernel::ROOT_RANK = 0;
const int MpiMatrixKernel::BLOCK_SIZE = 10;

MpiMatrixKernel::MpiMatrixKernel() :
        rank(MPI::COMM_WORLD.Get_rank()),
        groupSize(MPI::COMM_WORLD.Get_size())
{
}

void MpiMatrixKernel::startup(const vector<string>& arguments)
{
    if (rank != ROOT_RANK)
        return;

    left = MatrixHelper::readMatrixFrom(arguments[0]);
    right = MatrixHelper::readMatrixFrom(arguments[1]);

    if (left.columns() != right.rows())
        throw MismatchedMatricesException(left.columns(), right.rows());

    result = Matrix<float>(left.rows(), right.columns());
}

void MpiMatrixKernel::run()
{
    broadcastSizes();
    scatterMatrices();
    multiply();
    gatherResult();
}

void MpiMatrixKernel::shutdown(const string& outputFileName)
{
    if (rank != ROOT_RANK)
        return;

    MatrixHelper::writeMatrixTo(outputFileName, result);
}

void MpiMatrixKernel::broadcastSizes()
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

void MpiMatrixKernel::scatterMatrices()
{
}

void MpiMatrixKernel::multiply()
{
    if (rank != ROOT_RANK)
        return;

    for (size_t i=0; i<left.rows(); i++)
    {
        for (size_t j=0; j<right.columns(); j++)
        {
            result(i, j) = 0;

            for (size_t k=0; k<left.columns(); k++)
            {
                result(i, j) += left(i, k) * right(k, j);
            }
        }
    }
}

void MpiMatrixKernel::gatherResult()
{
}
