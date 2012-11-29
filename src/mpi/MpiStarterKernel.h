#pragma once

#include "benchmark/Arguments.h"
#include "benchmark/BenchmarkKernel.h"
#include "mpi/MpiMatrixKernel.h"

#include <memory>

class MpiStarterKernel : public BenchmarkKernel
{
public:
    static bool wasCorrectlyStarted();

    MpiStarterKernel(const std::shared_ptr<BenchmarkKernel>& kernel);

    virtual std::size_t requiredInputs() const;
    virtual void startup(const Arguments &arguments);
    virtual void startup(const std::vector<std::string>& arguments);
    virtual void run();
    virtual void shutdown(const std::string& outputFilename);
    virtual bool statsShouldBePrinted() const;

private:
    static const char* const ENVIRONMENT_VARIABLE_NAME;

    std::shared_ptr<BenchmarkKernel> kernel;
};

inline std::size_t MpiStarterKernel::requiredInputs() const
{
    return kernel->requiredInputs();
}

inline bool MpiStarterKernel::statsShouldBePrinted() const
{
    return false;
}
