#pragma once

#include <vector>
#include <string>

class BenchmarkKernel
{
public:
    virtual ~BenchmarkKernel();
    virtual void startup(const std::vector<std::string>& arguments) = 0;
    virtual void run() = 0;
    virtual void shutdown(const std::string& outputFilename) = 0;
};

inline BenchmarkKernel::~BenchmarkKernel()
{
}
