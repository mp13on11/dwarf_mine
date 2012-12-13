#pragma once

#include "Matrix.h"

#include <istream>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <utility>

class MatrixHelper
{
public:
    static Matrix<float> readMatrixFrom(const std::string& fileName);
    static Matrix<float> readMatrixFrom(std::istream& stream);
    static std::pair<Matrix<float>, Matrix<float>> readMatrixPairFrom(std::istream& stream);
    static void writeMatrixTo(const std::string& fileName, const Matrix<float>& matrix);
    static void writeMatrixTo(std::ostream& stream, const Matrix<float>& matrix);
    static void fill(Matrix<float>& matrix, const std::function<float()>& generator);
private:
    static void fillMatrixFromStream(Matrix<float>& matrix, std::istream& stream);
    static std::vector<float> getValuesIn(const std::string& line);
};
