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

#pragma once


#include <iosfwd>
#include <memory>
#include <string>
#include <vector>
#include <functional>

template<typename T>
class Matrix;
class Communicator;

namespace MatrixHelper
{
    typedef std::pair<Matrix<float>, Matrix<float>> MatrixPair;
 
    size_t receiveWorkQueueSize(
        const Communicator& communicator,
        const int node,
        const int tag = 0);
    void sendWorkQueueSize(
        const Communicator& communicator,
        const int node,
        const size_t workQueueSize,
        const int tag = 0);
    void requestTransaction(
        const Communicator& communicator,
        const int sourceNode,
        const int targetNode,
        const int tag = 0);
    int waitForTransactionRequest(
        const Communicator& communicator,
        const int tag = 0);
    MatrixPair getNextWork(
        const Communicator& communicator,
        const int node,
        const int tag = 0);
    void sendNextWork(
        const Communicator& communicator,
        const MatrixPair& work,
        const int node,
        const int tag = 0);
    void isendNextWork(
        const Communicator& communicator,
        const MatrixPair& work,
        const int node,
        const int tag = 0);

    void sendMatrixTo(const Communicator& communicator, const Matrix<float>& matrix, const int node, const int tag = 0);
    void isendMatrixTo(const Communicator& communicator, const Matrix<float>& matrix, const int node, const int tag = 0);
    Matrix<float> receiveMatrixFrom(const Communicator& communicator, const int node, const int tag = 0);

    Matrix<float> readMatrixFrom(const std::string& fileName, bool binary = true);
    Matrix<float> readMatrixFrom(std::istream& stream);
    void writeMatrixPairTo(std::ostream& output, const MatrixPair& matrices);
    MatrixPair readMatrixPairFrom(std::istream& stream);
    void writeMatrixTo(const std::string& fileName, const Matrix<float>& matrix, bool binary = true);
    void writeMatrixTo(std::ostream& stream, const Matrix<float>& matrix);

    Matrix<float> readMatrixTextFrom(std::istream& stream);
    void writeMatrixTextTo(std::ostream& stream, const Matrix<float>& matrix);

    void fill(Matrix<float>& matrix, const std::function<float()>& generator);
    void validateMultiplicationPossible(const Matrix<float>& a, const Matrix<float>& b);
}
