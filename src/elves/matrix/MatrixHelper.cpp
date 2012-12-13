#include "MatrixHelper.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <iostream>
#include <sstream>

using namespace std;

void MatrixHelper::writeMatrixTo(const string& filename, const Matrix<float>& matrix)
{
    ofstream file;
    file.exceptions(ios_base::failbit);
    file.open(filename);

    writeMatrixTo(file, matrix);
}

void MatrixHelper::writeMatrixTo(ostream& output, const Matrix<float>& matrix)
{
    output << matrix.rows() << " " << matrix.columns() << endl;

    for (size_t i=0; i<matrix.rows(); i++)
    {
        for (size_t j=0; j<matrix.columns(); j++)
        {
            if(j>0)
               output << " "; 
            output << matrix(i, j);
        }
        output << endl;
    }
}

Matrix<float> MatrixHelper::readMatrixFrom(istream& stream)
{
    try
    {
        size_t rows, columns;
        stream >> rows;
        stream >> columns;
        string line;
        getline(stream, line);
        Matrix<float> matrix(rows, columns);
        fillMatrixFromStream(matrix, stream);
        return matrix;
    }
    catch (...)
    {
        cerr << "ERROR: Wrong matrices stream format." << endl;
        throw;
    }
}

Matrix<float> MatrixHelper::readMatrixFrom(const string& fileName)
{
    ifstream file;
    file.exceptions(ios_base::failbit);
    file.open(fileName);
    return readMatrixFrom(file);
}

pair<Matrix<float>, Matrix<float>> MatrixHelper::readMatrixPairFrom(std::istream& stream)
{
    pair<Matrix<float>, Matrix<float>> matrices;
    matrices.first = readMatrixFrom(stream);
    matrices.second = readMatrixFrom(stream);
    return matrices;
}

void MatrixHelper::fillMatrixFromStream(Matrix<float>& matrix, istream& stream)
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

vector<float> MatrixHelper::getValuesIn(const string& line)
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

void MatrixHelper::fill(Matrix<float>& matrix, const function<float()>& generator)
{
    for(size_t y = 0; y<matrix.rows(); y++)
    {
        for(size_t x = 0; x<matrix.columns(); x++)
        {
            matrix(y, x) = generator();
        }
    }
}
