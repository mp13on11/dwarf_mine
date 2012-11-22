#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>
#include "Matrix.h"

using namespace std;

template<typename T>
Matrix<T> readMatrix(const string &fileName);

template<typename T>
void fillMatrixFromStream(Matrix<T> &matrix, istream &stream);

template<typename T>
vector<T> getValuesIn(const string &line);

int main(int argc, const char *argv[])
{
	if (argc < 2)
	{
		cerr << "Usage: " << argv[0] << " <matrix file name>" << endl;
		return 1;
	}

	vector<string> arguments(argv + 1, argv + argc);

	try
	{
		Matrix<float> matrix = readMatrix<float>(arguments[0]);

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
		return 1;
	}

	Matrix<int> matrix(5, 5);
}

template<typename T>
Matrix<T> readMatrix(const string &fileName)
{
	ifstream file;
	file.exceptions(fstream::failbit);
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
