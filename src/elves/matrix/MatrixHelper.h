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

    void sendMatrixTo(const Communicator& communicator, const Matrix<float>& matrix, const int node, const int tag = 0);
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
