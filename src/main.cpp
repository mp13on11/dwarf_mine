#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>
#include "Matrix.h"

using namespace std;

void generateMatrix(const string &fileName);
void printMatrix(const string &fileName);

template<typename T>
Matrix<T> readMatrix(const string &fileName);

template<typename T>
void fillMatrixFromStream(Matrix<T> &matrix, istream &stream);

template<typename T>
vector<T> getValuesIn(const string &line);

void printUsage(const string &program);

int main(int argc, const char *argv[])
{
	if (argc < 3)
	{
		printUsage(argv[0]);
		return 1;
	}

	vector<string> arguments(argv + 1, argv + argc);

	if (arguments[0] == "generate")
	{
		generateMatrix(arguments[1]);
	}
	else if (arguments[0] == "read")
	{
		printMatrix(arguments[1]);
	}
	else
	{
		printUsage(argv[0]);
		return 1;
	}
}

void generateMatrix(const string &fileName)
{
	const size_t size = 1000;
	default_random_engine engine;
	uniform_real_distribution<float> distribution(0, 1000);

	ofstream file;
	file.exceptions(ios_base::failbit);
	file.open(fileName);

	file << size << " " << size << endl;

	for (size_t i=0; i<size; i++)
	{
		if (i % 100 == 0)
			cout << "written " << i << " rows" << endl;
		for (size_t j=0; j<size; j++)
		{
			file << " " << distribution(engine);
		}

		file << endl;
	}
}

void printMatrix(const string &fileName)
{
	try
	{
		Matrix<float> matrix = readMatrix<float>(fileName);

		for (size_t i=0; i<matrix.rows(); i++)
		{
			for (size_t j=0; j<matrix.columns(); j++)
			{
				cout << " " << setw(3) << matrix(i, j);
			}

			cout << endl;
		}
	}
	catch (exception &e)
	{
		cerr << "Fooooooo:" << e.what() << endl;
	}
}

template<typename T>
Matrix<T> readMatrix(const string &fileName)
{
	ifstream file;
	file.exceptions(ios_base::failbit);
	file.open(fileName);

	size_t rows, columns;
	file >> rows;
	file >> columns;
	string line;
	getline(file, line);
	Matrix<T> result(rows, columns);

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

template<typename T>
void fillMatrixFromStream(Matrix<T> &matrix, istream &stream)
{

	for (size_t i=0; i<matrix.rows() && stream.good(); i++)
	{
		string line;
		getline(stream, line);
		vector<T> values = getValuesIn<T>(line);

		for (size_t j=0; j<matrix.columns() && j<values.size(); j++)
			matrix(i, j) = values[j];
	}
}

template<typename T>
vector<T> getValuesIn(const string &line)
{
	istringstream stream(line);
	vector<T> result;
	copy(
			istream_iterator<T>(stream),
			istream_iterator<T>(),
			back_inserter<vector<T>>(result)
		);

	return result;
}

void printUsage(const string &program)
{
	cerr << "Usage: " << program << " <generate|read> <matrix file name>" << endl;
}
