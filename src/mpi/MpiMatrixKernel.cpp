#include "mpi/MpiMatrixKernel.h"
#include "tools/MismatchedMatricesException.h"
#include "tools/MatrixHelper.h"

#include <mpi.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <exception>
#include <cmath>

using namespace std;

const int MpiMatrixKernel::ROOT_RANK = 0;
const size_t MpiMatrixKernel::BLOCK_ROWS = 3;
const size_t MpiMatrixKernel::BLOCK_COLUMNS = 3;

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
    if (!isRoot())
        return;

    MatrixHelper::writeMatrixTo(outputFileName, result);
}

void MpiMatrixKernel::broadcastSizes()
{
	size_t sizes[4] = {
			BLOCK_ROWS, left.columns(),
			right.rows(), BLOCK_COLUMNS
	};
    MPI::COMM_WORLD.Bcast(sizes, 4 * sizeof(size_t), MPI::CHAR, ROOT_RANK);

    if (!isRoot())
    {
        left = Matrix<float>(sizes[0], sizes[1]);
        right = Matrix<float>(sizes[2], sizes[3]);
        result = Matrix<float>(sizes[0], sizes[3]);
    }
}

vector<float> MpiMatrixKernel::distributeBuffer(const float* sendBuffer, const size_t rows, const size_t columns)
{
	size_t size = rows * columns;
	float* recieveBufferTemp = new float[size];
	if (MPI_Scatter(
			const_cast<float*>(sendBuffer), size, MPI_FLOAT,
			recieveBufferTemp, size, MPI_FLOAT,
			ROOT_RANK,
			MPI::COMM_WORLD
			) < 0)
	{
		throw new exception();
	}
	vector<float> recieveBuffer;
	recieveBuffer.assign(recieveBufferTemp, recieveBufferTemp + size);
	delete[] recieveBufferTemp;
	return recieveBuffer;
}

vector<float> MpiMatrixKernel::transposeMatrix(const float* data, const size_t rows, const size_t columns)
{
	vector<float> buffer;
	for (size_t column = 0; column < columns; ++column)
	{
		for (size_t row = 0; row < rows; ++row)
		{
			buffer.push_back(data[columns*row + column]);
		}
	}
	return buffer;
}

void MpiMatrixKernel::scatterMatrices()
{
    vector<float> leftBuffer = distributeBuffer(left.buffer(), BLOCK_ROWS, left.columns());

    vector<float> rightSendingBuffer;
    if (isRoot())
	{
    	rightSendingBuffer = transposeMatrix(right.buffer(), right.rows(), right.columns());
	}
    vector<float> rightRecievingBuffer = distributeBuffer(rightSendingBuffer.data(), BLOCK_COLUMNS, right.rows());
    vector<float> rightBuffer = transposeMatrix(rightRecievingBuffer.data(), rightRecievingBuffer.size() / BLOCK_COLUMNS, BLOCK_COLUMNS);

    left = Matrix<float>(BLOCK_ROWS, leftBuffer.size() / BLOCK_ROWS, move(leftBuffer));
	right = Matrix<float>(BLOCK_COLUMNS, rightBuffer.size() / BLOCK_COLUMNS, move(rightBuffer));
}

void MpiMatrixKernel::multiply()
{
	temp = Matrix<float>(left.rows(), right.columns());
    for (size_t i=0; i<left.rows(); i++)
    {
        for (size_t j=0; j<right.columns(); j++)
        {
        	temp(i, j) = 0;
            for (size_t k=0; k<left.columns(); k++)
            {
            	temp(i, j) += left(i, k) * right(k, j);
            }
        }
    }
}

void gather(float* sending, Matrix<float>& s, float* recieving, Matrix<float>& r)
{
	MPI_Gather(
			sending, s.rows()* s.columns(), MPI_FLOAT,
			recieving, r.rows() * r.columns(), MPI_FLOAT,
			0,
			MPI::COMM_WORLD
		);
}

void MpiMatrixKernel::gatherResult()
{
	float* recievingBuffer;
	if (isRoot())
	{
		recievingBuffer = new float[result.rows() * result.columns()];
	}
	gather(const_cast<float*>(temp.buffer()), temp, recievingBuffer, result);

	if (isRoot())
		delete[] recievingBuffer;
//	if (isRoot())
//	{
//		cout << "RANK "<<rank<< " before convert"<<endl;
////		vector<float> recieveBuffer;
////		recieveBuffer.assign(recievingBuffer, recievingBuffer + result.rows() * result.columns());
////		for (size_t i = 0; i< result.rows() * result.columns(); i++)
////		{
////			cout<<" "<<i<< "  "<<recievingBuffer[i]<<endl;
////		}
////		cout <<endl;
//		cout << "RANK "<<rank<< " before delete[]"<<endl;
//		//delete recievingBuffer	;
//		cout << "RANK "<<rank<< " after delete[]"<<endl;
//
//	}

}
