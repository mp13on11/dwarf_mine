#pragma once

#include "Matrix.h"

#include <istream>
#include <memory>
#include <string>
#include <vector>
#include <functional>

class MatrixHelper
{
public:
    static Matrix<float> readMatrixFrom(std::istream& input);
    static void writeMatrixTo(std::ostream& output, const Matrix<float>& matrix);
    static void fill(Matrix<float>& matrix, const std::function<float()>& generator);

private:
    static void fillMatrixFromStream(Matrix<float>& matrix, std::istream& stream);
    static std::vector<float> getValuesIn(const std::string& line);
};
