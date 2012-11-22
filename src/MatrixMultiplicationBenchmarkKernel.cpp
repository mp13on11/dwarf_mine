#include "MatrixMultiplicationBenchmarkKernel.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <iterator>
#include <iostream>
#include <sstream>

using namespace std;

class UnknownNameException : public exception
{
public:
	UnknownNameException(const string &name)
		: message("Unknown matrix multiplication benchmark name: ")
	{
		message += name;
	}

	virtual ~UnknownNameException() throw()
	{
	}

	virtual const char* what() const throw()
	{
		return message.c_str();
	}

private:
	string message;
};

unique_ptr<MatrixMultiplicationBenchmarkKernel> MatrixMultiplicationBenchmarkKernel::create(const string &name)
{
	if (name == "cuda")
		// TODO: create a Cuda kernel here
		return unique_ptr<MatrixMultiplicationBenchmarkKernel>();
	else if (name == "mpi")
		// TODO: create an MPI kernel here
		return unique_ptr<MatrixMultiplicationBenchmarkKernel>();
	else if (name == "smp")
		// TODO: create an SMP kernel here
		return unique_ptr<MatrixMultiplicationBenchmarkKernel>();
	else
		throw UnknownNameException(name);
}

void MatrixMultiplicationBenchmarkKernel::writeMatrixTo(const string &filename, const Matrix<float> &matrix)
{
	ofstream file;
	file.exceptions(ios_base::failbit);
	file.open(filename);

	file << matrix.rows() << " " << matrix.columns() << endl;

	for (size_t i=0; i<matrix.rows(); i++)
	{
		for (size_t j=0; j<matrix.columns(); j++)
		{
			file << " " << matrix(i, j);
		}

		file << endl;
	}
}

Matrix<float> MatrixMultiplicationBenchmarkKernel::readMatrixFrom(const string &fileName)
{
	ifstream file;
	file.exceptions(ios_base::failbit);
	file.open(fileName);

	size_t rows, columns;
	file >> rows;
	file >> columns;
	string line;
	getline(file, line);
	Matrix<float> result(rows, columns);

	try
	{
		fillMatrixFromStream(result, file);
	}
	catch (...)
	{
		cerr << "Warning. Missing line..." << endl;
	}

	return result;
}

void MatrixMultiplicationBenchmarkKernel::fillMatrixFromStream(Matrix<float> &matrix, istream &stream)
{
	for (size_t i=0; i<matrix.rows() && stream.good(); i++)
	{
		string line;
		getline(stream, line);
		vector<float> values = getValuesIn(line);

		for (size_t j=0; j<matrix.columns() && j<values.size(); j++)
			matrix(i, j) = values[j];
	}
}

vector<float> MatrixMultiplicationBenchmarkKernel::getValuesIn(const string &line)
{
	istringstream stream(line);
	vector<float> result;
	copy(
			istream_iterator<float>(stream),
			istream_iterator<float>(),
			back_inserter<vector<float>>(result)
		);

	return result;
}
