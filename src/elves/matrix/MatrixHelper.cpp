#include "MatrixHelper.h"
#include "Matrix.h"
#include "MismatchedMatricesException.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <mpi.h>

using namespace std;

namespace MatrixHelper
{
    void sendMatrixTo(const Matrix<float>& matrix, NodeId node)
    {
        unsigned long dimensions[2] =
        {
            matrix.rows(),
            matrix.columns()
        };
        auto numElements = dimensions[0] * dimensions[1];
        MPI::COMM_WORLD.Send(dimensions, 2, MPI::UNSIGNED_LONG, node, 0);
        MPI::COMM_WORLD.Send(matrix.buffer(), numElements, MPI::FLOAT, node, 0);
    }

    Matrix<float> receiveMatrixFrom(NodeId node)
    {
        unsigned long dimensions[2];
        MPI::COMM_WORLD.Recv(dimensions, 2, MPI::UNSIGNED_LONG, node, 0);
        auto rows = dimensions[0];
        auto cols = dimensions[1];
        Matrix<float> result(rows, cols);

        MPI::COMM_WORLD.Recv(result.buffer(), rows*cols, MPI::FLOAT, node, 0);
        return result;
    }

    void writeMatrixTo(const string& filename, const Matrix<float>& matrix)
    {
        ofstream file;
        file.open(filename, ios_base::binary);

        if (!file.is_open())
            throw runtime_error("Failed to open matrix file for write: " + filename);

        writeMatrixTo(file, matrix);
    }

    void writeMatrixTo(ostream& output, const Matrix<float>& matrix)
    {
        size_t dimensions[] = { matrix.rows(), matrix.columns() };
        output.write(reinterpret_cast<const char*>(dimensions), sizeof(size_t)*2);
        output.write(reinterpret_cast<const char*>(matrix.buffer()), sizeof(float)*matrix.rows()*matrix.columns());

        if (output.bad())
            throw runtime_error("Failed to write matrix to stream in " + string(__FILE__));
    }

    void writeMatrixPairTo(ostream& output, const pair<Matrix<float>, Matrix<float>>& matrices)
    {
        writeMatrixTo(output, matrices.first);
        writeMatrixTo(output, matrices.second);
    }

    Matrix<float> readMatrixFrom(istream& stream)
    {
        try
        {
            size_t dimensions[2];
            stream.read(reinterpret_cast<char*>(dimensions), sizeof(size_t)*2);

            size_t rows(dimensions[0]);
            size_t columns(dimensions[1]);

            Matrix<float> matrix(rows, columns);
            stream.read(reinterpret_cast<char*>(matrix.buffer()), sizeof(float) * rows * columns);

            if (stream.bad() || stream.fail())
                throw runtime_error("Failed to read matrix size from stream in " + string(__FILE__));

            return matrix;
        }
        catch (...)
        {
            cerr << "ERROR: Wrong matrices stream format." << endl;
            throw;
        }
    }

    Matrix<float> readMatrixFrom(const string& filename)
    {
        ifstream file;
        file.open(filename, ios_base::binary);

        if (!file.is_open())
            throw runtime_error("Failed to open matrix file for read: " + filename);

        return readMatrixFrom(file);
    }

    pair<Matrix<float>, Matrix<float>> readMatrixPairFrom(std::istream& stream)
    {
        pair<Matrix<float>, Matrix<float>> matrices;
        matrices.first = readMatrixFrom(stream);
        matrices.second = readMatrixFrom(stream);
        return matrices;
    }

    void validateMultiplicationPossible(const Matrix<float>& a, const Matrix<float>& b)
    {
        if (a.columns() != b.rows())
            throw MismatchedMatricesException(a.columns(), b.rows());
    }

    void fill(Matrix<float>& matrix, const function<float()>& generator)
    {
        for(size_t y = 0; y<matrix.rows(); y++)
        {
            for(size_t x = 0; x<matrix.columns(); x++)
            {
                matrix(y, x) = generator();
            }
        }
    }
}
