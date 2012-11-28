#pragma once

#include "benchmark/BenchmarkKernel.h"
#include "tools/Matrix.h"


class MpiMatrixKernel : public BenchmarkKernel
{
public:
    MpiMatrixKernel();
    virtual std::size_t requiredInputs() const;
    virtual void startup(const std::vector<std::string>& arguments);
    virtual void run();
    virtual void shutdown(const std::string& outputFilename);

private:
    static const int ROOT_RANK;

    const int rank;
    const int groupSize;
    size_t blockRows;
    size_t blockColumns;
    Matrix<float> left;
    Matrix<float> right;
    Matrix<float> result;
    int leftRows;
    int leftColumns;
    int rightRows;
    int rightColumns;


    void broadcastSizes();
    void scatterMatrices();
    std::vector<float> scatterBuffer(const float* buffer, size_t bufferSize, size_t chunkSize);
    std::vector<float> changeOrder(const float* matrix, size_t rows, size_t columns);
    void multiply();
    void gatherResult();
    bool isRoot() const;
};

inline bool MpiMatrixKernel::isRoot() const
{
    return rank == ROOT_RANK;
}

inline std::size_t MpiMatrixKernel::requiredInputs() const
{
    return 2;
}
