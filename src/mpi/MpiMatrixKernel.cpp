#include "mpi/MpiMatrixKernel.h"
#include "tools/MismatchedMatricesException.h"
#include "tools/MatrixHelper.h"

#include <mpi.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <exception>
#include <cmath>
#include <utility>

#ifdef __ECLIPSE_DEVELOPMENT__
	// small fix for Eclipse which can not detect the move method - during compiling using make
	// outside eclipse, this symbol will never be defined
	template<typename T>
	T&& move(T& value);
#endif

using namespace std;

const int MpiMatrixKernel::ROOT_RANK = 0;
const size_t MpiMatrixKernel::BLOCK_ROWS = 2;
const size_t MpiMatrixKernel::BLOCK_COLUMNS = 4;

MpiMatrixKernel::MpiMatrixKernel() :
        rank(MPI::COMM_WORLD.Get_rank()),
        groupSize(MPI::COMM_WORLD.Get_size())
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
    cout<< rank << " - left["<<left.rows()<<", "<<left.columns()<<"]"<< " - right["<<right.rows()<<", "<<right.columns()<<"]"<<endl;
    cout<< rank << " - right["<<right.rows()<<", "<<right.columns()<<"]"<< " - right["<<right.rows()<<", "<<right.columns()<<"]"<<endl;
}

vector<float> MpiMatrixKernel::distributeBuffer(const float* sendBuffer, const size_t rows, const size_t columns, string message)
{
	size_t size = (rows)* (columns);
	stringstream s;
	s<<message << "  ";
	if (rank == ROOT_RANK)
	for (int i = 0; i < size; i++)
	{
		s <<sendBuffer[i]<<" ";
	}
	cout << "SEND     "<<rank<<" - "<<s.str() <<endl<<endl;
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
	for (int i = 0; i < size; i++)
	{
		s <<"["<<rank<<", "<<recieveBufferTemp[i]<<"] ";
	}
	cout << "RECIEVED "<<rank<<" - "<<s.str() <<endl<<endl;
	vector<float> recieveBuffer;
	recieveBuffer.assign(recieveBufferTemp, recieveBufferTemp + size);
	delete[] recieveBufferTemp;
	return recieveBuffer;
}

vector<float> MpiMatrixKernel::transposeMatrix(const float* data, const size_t rows, const size_t columns, stringstream& out)
{
	vector<float> buffer;
	out << "TRANSPOSE  ";
	for (size_t i = 0; i < rows * columns; ++i)
	{
		out << data[i]<< " ";
	}
	out <<endl;
	for (size_t column = 0; column < columns; ++column)
	{
		for (size_t row = 0; row < rows; ++row)
		{
			buffer.push_back(data[columns * row + column]);
		}
	}
	out << "TRANSPOSED ";
	for (size_t i = 0; i < rows * columns; ++i)
	{
		out << buffer[i]<< " ";
	}
	out <<endl;
	return buffer;
}

void MpiMatrixKernel::scatterMatrices()
{
    vector<float> leftBuffer = distributeBuffer(left.buffer(), BLOCK_ROWS, left.columns(), "LEFT");
	stringstream out;
    out << "BEFORE TRANSPOSE "<<rank<<endl;
    vector<float> rightSendingBuffer;
    if (isRoot())
	{
    	rightSendingBuffer = transposeMatrix(right.buffer(), right.rows(), right.columns(), out);
	}

	vector<float> recieveBuffer;
    if (isRoot())
    {
		for (int slaveRank = 1; slaveRank < groupSize; ++slaveRank)
		{
			MPI_Send(
				const_cast<float*>(rightSendingBuffer.data()), rightSendingBuffer.size(), MPI_FLOAT,
				slaveRank, 0, MPI_COMM_WORLD);
		}
    }
    else
    {
    	int size = right.rows()*right.columns();
    	float* recieveBufferTemp = new float[size];
    	MPI_Recv(
    			recieveBufferTemp, size, MPI_FLOAT,
    			ROOT_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		recieveBuffer.assign(recieveBufferTemp, recieveBufferTemp + size);
		delete[] recieveBufferTemp;
    }

    //vector<float> rightRecievingBuffer = distributeBuffer(rightSendingBuffer.data(), BLOCK_COLUMNS, right.rows(), "RIGHT");
    out << "AFTER TRANSPOSE "<<rank<<endl;
    //vector<float> rightBuffer = transposeMatrix(rightRecievingBuffer.data(), rightRecievingBuffer.size() / BLOCK_COLUMNS, BLOCK_COLUMNS, out);
    vector<float> rightBuffer = recieveBuffer;

    left = Matrix<float>(BLOCK_ROWS, leftBuffer.size() / BLOCK_ROWS, move(leftBuffer));
	right = Matrix<float>(BLOCK_COLUMNS, rightBuffer.size() / BLOCK_COLUMNS, move(rightBuffer));

	out << "LEFT "<<rank<<endl;
	for (size_t r = 0; r < left.rows(); r++)
	{
		for (size_t c = 0; c < left.columns(); c++)
		{
			out<<left(r,c)<<" ";
		}
		out<<endl;
	}
	out << endl<<endl<< "RIGHT "<<rank<<endl;
	for (size_t r = 0; r < right.rows(); r++)
	{
		for (size_t c = 0; c < right.columns(); c++)
		{
			out<<right(r,c)<<" ";
		}
		out<<endl;
	}
	cout <<out.str()<<endl;
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

void MpiMatrixKernel::gatherResult()
{
	size_t recievingSize = result.rows() * result.columns();
	float* recievingBuffer = nullptr;
	if (isRoot())
	{
		recievingBuffer = new float[recievingSize];
	}

	MPI_Gather(const_cast<float*>(temp.buffer()), temp.rows() * temp.columns(), MPI::FLOAT,
					recievingBuffer, temp.rows() * temp.columns(), MPI::FLOAT,
					ROOT_RANK,
					MPI::COMM_WORLD);
	if (isRoot())
	{
		vector<float> recievedBuffer;
		recievedBuffer.assign(recievingBuffer, recievingBuffer + recievingSize);
		delete[] recievingBuffer;

		result = Matrix<float>(result.rows(), result.columns(), move(recievedBuffer));
	}
}
