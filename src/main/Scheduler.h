#pragma once

#include "BenchmarkResults.h"

struct ProblemStatement;

class Scheduler
{
public:
    Scheduler(const BenchmarkResult& benchmarkResult);
    virtual void dispatch(ProblemStatement& statement);
};
