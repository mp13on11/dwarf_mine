#include "MatrixHelper.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <iostream>
#include <sstream>

using namespace std;

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

Matrix<float> MatrixHelper::readMatrixFrom(istream& input)
{
    size_t rows, columns;
    input >> rows;
    input >> columns;
    string line;
    getline(input, line);
    Matrix<float> result(rows, columns);

    try
    {
        fillMatrixFromStream(result, input);
    }
    catch (...)
    {
        cerr << "Warning. Missing line..." << endl;
    }

    return result;
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
