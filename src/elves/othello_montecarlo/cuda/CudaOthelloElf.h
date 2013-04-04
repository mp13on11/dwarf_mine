#pragma once

#include <OthelloElf.h>
#include <OthelloUtil.h>
#include <State.h>

class CudaOthelloElf : public OthelloElf
{
public:
    virtual std::vector<Result> getMovesFor(State& state, size_t reiterations, size_t nodeId, size_t commonSeed);
};
