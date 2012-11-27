#pragma once

#include <benchmark/BenchmarkKernel.h>

class SMPMatrixKernel : public BenchmarkKernel
{

public:
    virtual void startup(const std::vector<std::string>& arguments);
    virtual void run();
    virtual void shutdown(const std::string& outputFilename);

private:

    std::size_t matrixARows;
    std::size_t matrixACols;
    std::size_t matrixBRows;
    std::size_t matrixBCols;

};