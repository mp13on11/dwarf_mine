#pragma once

#include "../benchmark-lib/BenchmarkKernel.h"

class CudaMatrixKernel : public BenchmarkKernel
{
public:
    virtual void startup(const std::vector<std::string>& arguments);
    virtual void run();
    virtual void shutdown(const std::string& outputFilename);
};
