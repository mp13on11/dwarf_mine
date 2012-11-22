#pragma once

#include <benchmark/BenchmarkKernel.h>
#include "Memory.h"
#include <memory>

class Cublas;

class CudaMatrixKernel : public BenchmarkKernel
{
public:
    virtual void startup(const std::vector<std::string>& arguments);
    virtual void run();
    virtual void shutdown(const std::string& outputFilename);

private:
    CudaUtils::Memory<float> matrixMemA;
    CudaUtils::Memory<float> matrixMemB;
    CudaUtils::Memory<float> outputMatrix;
    std::unique_ptr<Cublas> cublas;

    std::size_t matrixARows;
    std::size_t matrixACols;
    std::size_t matrixBCols;
};
