#pragma once

#include "BenchmarkKernel.h"
#include "Matrix.h"

#include <istream>
#include <memory>
#include <string>
#include <vector>

class MatrixMultiplicationBenchmarkKernel : public BenchmarkKernel
{
public:
    static std::unique_ptr<MatrixMultiplicationBenchmarkKernel> create(const std::string& name);

protected:
    static Matrix<float> readMatrixFrom(const std::string& fileName);
    static void writeMatrixTo(const std::string& fileName, const Matrix<float>& matrix);

private:
    static void fillMatrixFromStream(Matrix<float>& matrix, std::istream& stream);
    static std::vector<float> getValuesIn(const std::string& line);
};
