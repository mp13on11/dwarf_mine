/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

#include "MatrixHelper.h"
#include "common/Communicator.h"
#include "Matrix.h"
#include "MismatchedMatricesException.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

namespace MatrixHelper
{
    static void fillMatrixFromStream(Matrix<float>& matrix, istream& stream);
    static vector<float> getValuesIn(const string& line);

    size_t receiveWorkQueueSize(
        const Communicator& communicator,
        const int node,
        const int tag)
    {
        int size;
        communicator->Recv(&size, 1, MPI::INT, node, tag);
        return size_t(size);
    }

    void sendWorkQueueSize(
        const Communicator& communicator,
        const int node,
        const size_t workQueueSize,
        const int tag)
    {
        int size = int(workQueueSize);
        communicator->Send(&size, 1, MPI::INT, node, tag);
    }

    void requestTransaction(
        const Communicator& communicator,
        const int sourceNode,
        const int targetNode,
        const int tag)
    {
        communicator->Send(&sourceNode, 1, MPI::INT, targetNode, tag);
    }

    int waitForTransactionRequest(
        const Communicator& communicator,
        const int tag)
    {
        int node;
        communicator->Recv(&node, 1, MPI::INT, MPI::ANY_SOURCE, tag);
        return node;
    }

    MatrixPair getNextWork(
        const Communicator& communicator,
        const int node,
        const int tag)
    {
        Matrix<float> left = receiveMatrixFrom(communicator, node, tag);
        Matrix<float> right = receiveMatrixFrom(communicator, node, tag);
        return MatrixPair(move(left), move(right));
    }   
    
    void sendNextWork(
        const Communicator& communicator,
        const MatrixPair& work,
        const int node,
        const int tag)
    {
        sendMatrixTo(communicator, work.first, node, tag);
        sendMatrixTo(communicator, work.second, node, tag);
    }   
    
    void isendNextWork(
        const Communicator& communicator,
        const MatrixPair& work,
        const int node,
        const int tag)
    {
        isendMatrixTo(communicator, work.first, node, tag);
        isendMatrixTo(communicator, work.second, node, tag);
    }   

    void sendMatrixTo(
        const Communicator& communicator,
        const Matrix<float>& matrix,
        const int node,
        const int tag)
    {
        unsigned long dimensions[2] =
        {
            matrix.rows(),
            matrix.columns()
        };
        auto numElements = dimensions[0] * dimensions[1];
        communicator->Send(dimensions, 2, MPI::UNSIGNED_LONG, node, tag);
        communicator->Send(matrix.buffer(), numElements, MPI::FLOAT, node, tag);
    }

    void isendMatrixTo(
        const Communicator& communicator,
        const Matrix<float>& matrix,
        const int node,
        const int tag)
    {
        unsigned long dimensions[2] =
        {
            matrix.rows(),
            matrix.columns()
        };
        auto numElements = dimensions[0] * dimensions[1];
        communicator->Send(dimensions, 2, MPI::UNSIGNED_LONG, node, tag);
        communicator->Isend(matrix.buffer(), numElements, MPI::FLOAT, node, tag);
    }

    Matrix<float> receiveMatrixFrom(
        const Communicator& communicator,
        const int node,
        const int tag)
    {
        unsigned long dimensions[2];
        communicator->Recv(dimensions, 2, MPI::UNSIGNED_LONG, node, tag);
        auto rows = dimensions[0];
        auto cols = dimensions[1];
        Matrix<float> result(rows, cols);
        communicator->Recv(result.buffer(), rows*cols, MPI::FLOAT, node, tag);
        return result;
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
