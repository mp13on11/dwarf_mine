#pragma once

#include "common/MpiHelper.h"

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>
#include <functional>

template<typename T>
class Matrix;

namespace MatrixHelper
{
    typedef std::pair<Matrix<float>, Matrix<float>> MatrixPair;
    const int TAG_REQUEST_SLICE = 0;
 
    //
    // Send via MPI
    void requestNextSlice(NodeId node);
    NodeId getNextSliceRequest();
    void sendMatrixTo(const Matrix<float>& matrix, NodeId node);
    Matrix<float> receiveMatrixFrom(NodeId node);

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
