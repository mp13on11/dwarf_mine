#pragma once

#include "benchmark/Arguments.h"

#include <memory>
#include <string>
#include <vector>

class BenchmarkKernel
{
public:
    virtual ~BenchmarkKernel();
    virtual std::size_t requiredInputs() const = 0;
    virtual void startup(const Arguments& arguments);
    virtual void startup(const std::vector<std::string>& arguments) = 0;
    virtual void run() = 0;
    virtual void shutdown(const std::string& outputFilename) = 0;
    virtual bool statsShouldBePrinted() const;
};

extern std::shared_ptr<BenchmarkKernel> createKernel();

inline void BenchmarkKernel::startup(const Arguments& arguments)
{
    startup(arguments.inputFileNames());
}

inline bool BenchmarkKernel::statsShouldBePrinted() const
{
    return true;
}
