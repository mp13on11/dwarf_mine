#pragma once

#include "Configuration.h"
#include <iosfwd>

struct DataGenerationParameters;
class ProblemStatement;

class Scheduler
{
public:
    Scheduler();
    virtual ~Scheduler() = 0;

    void provideData(const ProblemStatement& problem);
    void outputData(const ProblemStatement& problem);

    virtual void configureWith(const Configuration& config);
    virtual void dispatch() = 0;

protected:
    virtual void generateData(const DataGenerationParameters& params) = 0;
    virtual void provideData(std::istream& input) = 0;
    virtual void outputData(std::ostream& output) = 0;
};

inline void Scheduler::configureWith(const Configuration&)
{
}
