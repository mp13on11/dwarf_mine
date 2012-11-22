#pragma once

#include <memory>
#include <string>
#include <vector>

class BenchmarkKernel
{
public:
    virtual ~BenchmarkKernel();
    virtual void startup(const std::vector<std::string>& arguments) = 0;
    virtual void run() = 0;
    virtual void shutdown(const std::string& outputFilename) = 0;
};

extern std::shared_ptr<BenchmarkKernel> createKernel();