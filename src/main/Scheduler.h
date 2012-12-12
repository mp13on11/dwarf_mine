#pragma once

#include "BenchmarkResults.h"

struct ProblemStatement;

class Scheduler
{
public:
    Scheduler(const BenchmarkResult& benchmarkResult);

    void dispatch(ProblemStatement& statement);
};
