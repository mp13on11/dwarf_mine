#pragma once

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>
#include <functional>

template<typename T>
class Matrix;

namespace MatrixHelper
{
    Matrix<float> readMatrixFrom(const std::string& fileName);
    Matrix<float> readMatrixFrom(std::istream& stream);
    void writeMatrixPairTo(std::ostream& output, const std::pair<Matrix<float>, Matrix<float>>& matrices);
    std::pair<Matrix<float>, Matrix<float>> readMatrixPairFrom(std::istream& stream);
    void writeMatrixTo(const std::string& fileName, const Matrix<float>& matrix);
    void writeMatrixTo(std::ostream& stream, const Matrix<float>& matrix);
    void fill(Matrix<float>& matrix, const std::function<float()>& generator);
    void validateMultiplicationPossible(const Matrix<float>& a, const Matrix<float>& b);
}
