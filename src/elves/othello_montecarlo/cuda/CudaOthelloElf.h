#pragma once

#include <OthelloElf.h>
#include <OthelloUtil.h>
#include <State.h>

class CudaOthelloElf : public OthelloElf
{
public:
    virtual Result getBestMoveFor(State& state, size_t reiterations, size_t nodeId, size_t commonSeed);
private:
	Result getBestMoveForSimple(State& state, size_t reiterations, size_t nodeId, size_t commonSeed);
};
