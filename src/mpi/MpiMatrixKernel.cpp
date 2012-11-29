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

MpiMatrixKernel::MpiMatrixKernel() :
        rank(MPI::COMM_WORLD.Get_rank()),
        groupSize(MPI::COMM_WORLD.Get_size()),
        blockRows(1),
        blockColumns(1),
        leftRows(0),
        leftColumns(0),
        rightRows(0),
        rightColumns(0)
{
}

void MpiMatrixKernel::startup(const vector<string>& arguments)
{
    if (rank != ROOT_RANK)
        return;

    left = MatrixHelper::readMatrixFrom((const string)(arguments[0]));
    right = MatrixHelper::readMatrixFrom((const string)(arguments[1]));

    if (left.columns() != right.rows())
        throw MismatchedMatricesException(left.columns(), right.rows());
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
    if (!isRoot())
        return;

    MatrixHelper::writeMatrixTo(outputFileName, result);
}

#include <unistd.h>

void MpiMatrixKernel::broadcastSizes()
{
    size_t sizes[4] = {
            left.rows(), left.columns(),
            right.rows(), right.columns()
    };
    MPI::COMM_WORLD.Bcast(sizes, 4 * sizeof(size_t), MPI::CHAR, ROOT_RANK);

    leftRows = sizes[0];
    leftColumns = sizes[1];
    rightRows = sizes[2];
    rightColumns = sizes[3];

    int columnBlocks = round(sqrt(groupSize * 1.0));
    int rowBlocks = groupSize / blockColumns;

    blockColumns = rightColumns / columnBlocks;
    blockRows = leftRows / rowBlocks;
}

vector<float> MpiMatrixKernel::scatterBuffer(const float* buffer, size_t bufferSize, size_t chunkSize)
{
    float* temp = new float[chunkSize];

    if (isRoot())
    {
        size_t currentChunkOffset = 0;
        size_t currentChunkSize = chunkSize;
        for (int workerRank = 0; workerRank < groupSize; ++workerRank)
        {
            MPI_Send(
                const_cast<float*>(buffer) + currentChunkOffset, currentChunkSize, MPI_FLOAT,
                workerRank, 0, MPI_COMM_WORLD);
            currentChunkOffset += chunkSize;
            currentChunkSize = min(bufferSize - ((workerRank + 1)* chunkSize), chunkSize);
            if (currentChunkOffset >= bufferSize)
            {
                currentChunkOffset = 0;
                currentChunkSize = chunkSize;
            }
        }
    }
    MPI_Recv(
        temp, chunkSize, MPI_FLOAT,
        ROOT_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    vector<float> recieveBuffer;
    recieveBuffer.assign(temp, temp + chunkSize);
    delete[] temp;
    return recieveBuffer;
}

vector<float> MpiMatrixKernel::changeOrder(const float* buffer, size_t rows, size_t columns)
{
    vector<float> ordered;
    for (size_t column = 0; column < columns; ++column)
    {
        for (size_t row = 0; row < rows; ++row)
        {
            ordered.push_back(buffer[columns * row + column]);
        }
    }
    return ordered;
}

void MpiMatrixKernel::scatterMatrices()
{
    vector<float> leftBuffer = scatterBuffer(left.buffer(), leftColumns * leftRows, blockRows * leftColumns);
    vector<float> rightBuffer;
    if (isRoot())
    {
        // change to column major
        rightBuffer = changeOrder(right.buffer(), rightRows, rightColumns);
    }
    rightBuffer = scatterBuffer(rightBuffer.data(), rightBuffer.size(), blockColumns * rightRows);
    // change back to row major
    rightBuffer = changeOrder(rightBuffer.data(), blockColumns, rightRows);

    left = Matrix<float>(blockRows, leftColumns, move(leftBuffer));
    right = Matrix<float>(rightRows, blockColumns, move(rightBuffer));
}

void MpiMatrixKernel::multiply()
{
    result = Matrix<float>(left.rows(), right.columns());
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

void printToStream(const Matrix<float>& matrix, string prefix,  ostream& out)
{
    out << prefix <<endl;
    for (size_t i = 0; i < matrix.rows(); ++i)
    {
        for (size_t j = 0; j < matrix.columns(); j++)
        {
            out << matrix(i, j)<< " ";
        }
        out <<endl;
    }
    out<<endl;
}

void MpiMatrixKernel::gatherResult()
{
//  stringstream out;
//  out<< "Rank: "<< rank <<endl;
    //printToStream(left, "Left", out);
    //printToStream(right, "Right", out);
//  printToStream(result, "Result", out);
//  cout<< endl<<out.str()<<endl;
//  out.clear();
    float* buffer = nullptr;
    if (isRoot())
    {
        buffer = new float[leftRows * rightColumns];
    }
    MPI_Gather(
        const_cast<float*>(result.buffer()), result.rows() * result.columns(), MPI::FLOAT,
        buffer, result.rows() * result.columns(), MPI::FLOAT,
        ROOT_RANK,
        MPI::COMM_WORLD);

    if (isRoot())
    {
        Matrix<float> gatheredResult = Matrix<float>(leftRows, rightColumns);
        for (int worker = 0; worker < groupSize; ++worker)
        {
            for (size_t blockRow = 0, gatheredRow = (worker * blockRows) % (leftRows);
                    blockRow < blockRows && gatheredRow < leftRows;
                    ++blockRow, gatheredRow = (worker * blockRows + blockRow)  % (leftRows))
            {
                for (size_t blockColumn = 0, gatheredColumn = worker * blockColumns % (rightColumns) ;
                        blockColumn < blockColumns && gatheredColumn < rightColumns;
                        ++blockColumn, gatheredColumn = (worker * blockColumns + blockColumn) % rightColumns)
                {
//                  out << ">>>"<<gatheredRow<<"/"<<leftColumns<<" - "<<gatheredColumn <<"/" <<rightColumns<<"\n"
//                      << "   "<<blockRow<<"/"<<blockRows<<" - "<<blockColumn<<"/"<<blockColumns<<" = "<<worker<<endl;
//                  out<< "[("<<worker<<"/"<<groupSize<<") "<<gatheredRow<<", "<<gatheredColumn<<"] "<<worker * blockRows * blockColumns + blockRow * blockRows + blockColumn<<" "<<buffer[worker * blockRows * blockColumns + blockRow * blockRows + blockColumn]<< "  ";
                    gatheredResult(gatheredRow, gatheredColumn) =
                            buffer[worker * blockRows * blockColumns + blockRow * blockRows + blockColumn];
                }
            }
        }
//      for (size_t row = 0; row < gatheredResult.rows(); ++row)
//      {
//          for (size_t column = 0; column < gatheredResult.columns(); ++column)
//          {
//              gatheredResult(row, column) = buffer[
//                   row / blockRows + row % blockRows
//                   column / blockColums + column & blockColumns]
//          }
//      }
        //  cout << out.str() <<endl;
//      out.clear();
        //stringstream out;
        //printToStream(gatheredResult, "Final", out);
        //cout<<out.str()<<endl;
        delete[] buffer;
        result = move(gatheredResult);
    }
}
