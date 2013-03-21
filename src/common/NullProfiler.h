#pragma once

#include "Profiler.h"

class NullProfiler : public Profiler
{
public:
    virtual void beginIterationBlock();
    virtual void beginIteration();
    virtual void endIteration();
    virtual void endIterationBlock();
};

inline void NullProfiler::beginIterationBlock()
{
}

inline void NullProfiler::beginIteration()
{
}

inline void NullProfiler::endIteration()
{
}

inline void NullProfiler::endIterationBlock()
{
}