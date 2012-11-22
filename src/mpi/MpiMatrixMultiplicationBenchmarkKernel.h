#pragma once

#include "MatrixMultiplicationBenchmarkKernel.h"

class MpiMatrixMultiplicationBenchmarkKernel : public MatrixMultiplicationBenchmarkKernel
{
public:
    MpiMatrixMultiplicationBenchmarkKernel();
    virtual void startup(const std::vector<std::string>& arguments);
    virtual void run();
    virtual void shutdown(const std::string& outputFilename);

private:
    static const int ROOT_RANK;

    const int rank;
    Matrix<float> left;
    Matrix<float> right;
    Matrix<float> result;

    void broadcastSizes();
    void scatterMatrices();
    void multiply();
    void gatherResult();
};
