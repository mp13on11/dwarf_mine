#include "mpi/MpiMatrixKernel.h"
#include "tools/MismatchedMatricesException.h"
#include "tools/MatrixHelper.h"

#include <mpi.h>
#include <algorithm>
#include <exception>
#include <cmath>

#include <sstream>
#include <iomanip>

#ifdef __ECLIPSE_DEVELOPMENT__
    // small fix for Eclipse which can not detect the move method -
    // during compiling using make outside eclipse, this symbol will never be defined
    template<typename T>
    T&& move(T& value);
#endif

using namespace std;

const int MpiMatrixKernel::ROOT_RANK = 0;
const size_t MpiMatrixKernel::BLOCK_SIZE = 10;

MpiMatrixKernel::MpiMatrixKernel() :
        rank(MPI::COMM_WORLD.Get_rank()),
        groupSize(MPI::COMM_WORLD.Get_size()),
        rowBuffer(nullptr),
        columnBuffer(nullptr),
        resultBuffer(nullptr)
{
}

MpiMatrixKernel::~MpiMatrixKernel()
{
    if (columnBuffer != nullptr)
        delete[] columnBuffer;
    if (rowBuffer != nullptr)
        delete[] rowBuffer;
    if (resultBuffer != nullptr)
        delete[] resultBuffer;
}

void MpiMatrixKernel::startup(const vector<string>& arguments)
{
    if (rank != ROOT_RANK)
        return;

    left = MatrixHelper::readMatrixFrom((const string)(arguments[0]));
    right = MatrixHelper::readMatrixFrom((const string)(arguments[1]));

    if (left.columns() != right.rows())
        throw MismatchedMatricesException(left.columns(), right.rows());

    result = Matrix<float>(left.rows(), right.columns());
}

void MpiMatrixKernel::run()
{
    broadcastSizes();

    size_t rounds = blockCount() / groupSize;
    if (blockCount() % groupSize != 0)
        rounds += 1;

    for (size_t i=0; i<rounds; i++)
    {
        scatterMatrices(i);
        multiply(i);
        gatherResult(i);
    }
}

void MpiMatrixKernel::shutdown(const string& outputFileName)
{
    if (!isRoot())
        return;

    MatrixHelper::writeMatrixTo(outputFileName, result);
}

void MpiMatrixKernel::broadcastSizes()
{
    size_t sizes[4] = {
            left.rows(), left.columns(),
            right.rows(), right.columns()
        };
    MPI::COMM_WORLD.Bcast(sizes, 4 * sizeof(size_t), MPI::CHAR, ROOT_RANK);

    size_t leftRows = sizes[0];
    size_t leftColumns = sizes[1];
    size_t rightRows = sizes[2];
    size_t rightColumns = sizes[3];

    sentRows = min(BLOCK_SIZE, leftRows);
    sentColumns = min(BLOCK_SIZE, rightColumns);
    fullRows = leftRows;
    fullColumns = rightColumns;

    if (isRoot())
    {
        rowBuffer = new float[groupSize * sentRows * leftColumns];
        columnBuffer = new float[groupSize * rightRows * sentColumns];
        resultBuffer = new float[groupSize * sentRows * sentColumns];
    }
    else
    {
        left = Matrix<float>(sentRows, leftColumns);
        right = Matrix<float>(rightRows, sentColumns);
        result = Matrix<float>(sentRows, sentColumns);
        rowBuffer = new float[sentRows * leftColumns];
        columnBuffer = new float[rightRows * sentColumns];
        // we can simply use the buffer from the resulting matrix to send
        // the results back to root.
        resultBuffer = NULL;
    }
}

void MpiMatrixKernel::scatterMatrices(size_t round)
{
    if (isRoot())
    {
        for (int rank=0; rank<groupSize && blockIndex(round, rank) < blockCount(); rank++)
        {
            size_t startRow = rowIndex(round, rank);
            size_t startColumn = columnIndex(round, rank);
            size_t rowOffset = sentRows * left.columns() * rank;
            size_t columnOffset = sentColumns * right.rows() * rank;

            // rows are transferred in row-major order
            for (size_t i=0; i<sentRows && i+startRow<left.rows(); i++)
                for (size_t j=0; j<left.columns(); j++)
                    rowBuffer[rowOffset + i*left.columns() + j] = left(i+startRow, j);

            // columns are transferred in column-major order
            for (size_t j=0; j<sentColumns && j+startColumn<right.columns(); j++)
                for (size_t i=0; i<right.rows(); i++)
                    columnBuffer[columnOffset + j*right.rows() + i] = right(i, j+startColumn);
        }
    }

    MPI::COMM_WORLD.Scatter(
            rowBuffer, sentRows * left.columns(), MPI::FLOAT,
            rowBuffer, sentRows * left.columns(), MPI::FLOAT, ROOT_RANK
        );
    MPI::COMM_WORLD.Scatter(
            columnBuffer, right.rows() * sentColumns, MPI::FLOAT,
            columnBuffer, right.rows() * sentColumns, MPI::FLOAT, ROOT_RANK
        );

    if (isRoot())
        return;

    for (size_t i=0; i<left.rows(); i++)
        for (size_t j=0; j<left.columns(); j++)
            left(i, j) = rowBuffer[i * left.columns() + j];

    for (size_t j=0; j<right.columns(); j++)
        for (size_t i=0; i<right.rows(); i++)
            right(i, j) = columnBuffer[j * right.rows() + i];
}

void MpiMatrixKernel::multiply(size_t round)
{
    size_t rowOffset = 0;
    size_t columnOffset = 0;

    if (isRoot())
    {
        rowOffset = rowIndex(round, rank);
        columnOffset = columnIndex(round, rank);
    }

    for (size_t i=rowOffset; i<rowOffset+sentRows && i<left.rows(); i++)
    {
        for (size_t j=columnOffset; j<columnOffset+sentColumns && j<right.columns(); j++)
        {
            result(i, j) = 0;
            for (size_t k=0; k<left.columns(); k++)
            {
                result(i, j) += left(i, k) * right(k, j);
            }
        }
    }
}

void MpiMatrixKernel::gatherResult(size_t round)
{
    MPI::COMM_WORLD.Gather(
            result.buffer(), sentRows * sentColumns, MPI::FLOAT,
            resultBuffer, sentRows * sentColumns, MPI::FLOAT, ROOT_RANK
        );

    // ignore root rank, as its result is already stored in the result matrix
    for (int rank = 1; rank < groupSize; rank++)
    {
        size_t startRow = rowIndex(round, rank);
        size_t startColumn = columnIndex(round, rank);
        size_t offset = rank * sentRows * sentColumns;

        for (size_t i=0; i<sentRows && i+startRow < result.rows(); i++)
        {
            for (size_t j=0; j<sentColumns && j+startColumn < result.columns(); j++)
            {
                result(i + startRow, j + startColumn) = resultBuffer[offset + i*sentColumns + j];
            }
        }
    }
}
