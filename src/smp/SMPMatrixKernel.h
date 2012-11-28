#pragma once

#include <benchmark/BenchmarkKernel.h>
#include <tools/Matrix.h>

class SMPMatrixKernel : public BenchmarkKernel
{

public:
    virtual std::size_t requiredInputs() const;
    virtual void startup(const std::vector<std::string>& arguments);
    virtual void run();
    virtual void shutdown(const std::string& outputFilename);

private:

    Matrix<float> matrixA;
    Matrix<float> matrixB;
    Matrix<float> matrixC;

    std::size_t matrixARows;
    std::size_t matrixACols;
    std::size_t matrixBRows;
    std::size_t matrixBCols;

};
