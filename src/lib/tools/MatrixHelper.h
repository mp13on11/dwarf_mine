#pragma once

#include "tools/Matrix.h"

#include <istream>
#include <memory>
#include <string>
#include <vector>

class MatrixHelper
{
public:
    static Matrix<float> readMatrixFrom(const std::string& fileName);
    static void writeMatrixTo(const std::string& fileName, const Matrix<float>& matrix);

private:
    static void fillMatrixFromStream(Matrix<float>& matrix, std::istream& stream);
    static std::vector<float> getValuesIn(const std::string& line);
};
