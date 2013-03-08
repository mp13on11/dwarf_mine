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
    static void fillMatrixFromStream(Matrix<float>& matrix, istream& stream);
    static vector<float> getValuesIn(const string& line);

    void requestNextSlice(NodeId node)
    {
        MPI::COMM_WORLD.Send(&node, 1, MPI::INT, 0, MatrixHelper::TAG_REQUEST_SLICE);
    }

    NodeId getNextSliceRequest()
    {
        int nodeId;
        MPI::COMM_WORLD.Recv(&nodeId, 1, MPI::INT, MPI::ANY_SOURCE, MatrixHelper::TAG_REQUEST_SLICE);
        return NodeId(nodeId);
    }

    void sendMatrixTo(const Matrix<float>& matrix, NodeId node)
    {
        cout << "Sending ";
        unsigned long dimensions[2] =
        {
            matrix.rows(),
            matrix.columns()
        };
        auto numElements = dimensions[0] * dimensions[1];
        MPI::COMM_WORLD.Send(dimensions, 2, MPI::UNSIGNED_LONG, node, 0);
        MPI::COMM_WORLD.Send(matrix.buffer(), numElements, MPI::FLOAT, node, 0);
        cout << "done" << endl;
    }

    Matrix<float> receiveMatrixFrom(NodeId node)
    {
        cout << "Receiving ";
        unsigned long dimensions[2];
        MPI::COMM_WORLD.Recv(dimensions, 2, MPI::UNSIGNED_LONG, node, 0);
        auto rows = dimensions[0];
        auto cols = dimensions[1];
        Matrix<float> result(rows, cols);
        MPI::COMM_WORLD.Recv(result.buffer(), rows*cols, MPI::FLOAT, node, 0);
        return result;
        cout << "done" << endl;
    }

    void writeMatrixTo(const string& filename, const Matrix<float>& matrix, bool binary)
    {
        ofstream file;

        if (binary)
            file.open(filename, ios_base::binary);
        else
            file.open(filename);

        if (!file.is_open())
            throw runtime_error("Failed to open matrix file for write: " + filename);

        if (binary)
            writeMatrixTo(file, matrix);
        else
            writeMatrixTextTo(file, matrix);
    }

    void writeMatrixTo(ostream& output, const Matrix<float>& matrix)
    {
        uint64_t dimensions[] = { matrix.rows(), matrix.columns() };
        output.write(reinterpret_cast<const char*>(dimensions), sizeof(dimensions));
        output.write(reinterpret_cast<const char*>(matrix.buffer()), sizeof(float)*matrix.rows()*matrix.columns());

        if (output.bad())
            throw runtime_error("Failed to write matrix to stream in " + string(__FILE__));
    }

    void writeMatrixTextTo(ostream& output, const Matrix<float>& matrix)
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

            if (output.bad())
                throw runtime_error("Failed to write matrix to stream in " + string(__FILE__));
        }
    }

    Matrix<float> readMatrixTextFrom(istream& stream)
    {
        try
        {
            size_t rows, columns;
            stream >> rows;
            stream >> columns;

            if (stream.bad() || stream.fail())
            {
                throw runtime_error("Failed to read matrix size from stream in " + string(__FILE__));
            }
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

    void writeMatrixPairTo(ostream& output, const pair<Matrix<float>, Matrix<float>>& matrices)
    {
        writeMatrixTo(output, matrices.first);
        writeMatrixTo(output, matrices.second);
    }

    Matrix<float> readMatrixFrom(istream& stream)
    {
        try
        {
            uint64_t dimensions[2];
            stream.read(reinterpret_cast<char*>(dimensions), sizeof(dimensions));

            uint64_t rows(dimensions[0]);
            uint64_t columns(dimensions[1]);

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

    Matrix<float> readMatrixFrom(const string& filename, bool binary)
    {
        ifstream file;

        if (binary)
            file.open(filename, ios_base::binary);
        else
            file.open(filename);

        if (!file.is_open())
            throw runtime_error("Failed to open matrix file for read: " + filename);

        if (binary)
            return readMatrixFrom(file);
        else
            return readMatrixTextFrom(file);
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

    void fillMatrixFromStream(Matrix<float>& matrix, istream& stream)
    {
        for (size_t i=0; i<matrix.rows() && stream.good(); i++)
        {
            string line;
            getline(stream, line);

            if (stream.bad())
                throw runtime_error("Failed to read matrix from stream in " + string(__FILE__));

            vector<float> values = getValuesIn(line);

            for (size_t j=0; j<matrix.columns() && j<values.size(); j++)
                matrix(i, j) = values[j];
        }
    }

    vector<float> getValuesIn(const string& line)
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
}
