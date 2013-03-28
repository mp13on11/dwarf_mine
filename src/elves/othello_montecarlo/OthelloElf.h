#pragma once

#include <Elf.h>
#include <State.h>
#include <Result.h>

class OthelloElf : public Elf
{
public:
    virtual Result getBestMoveFor(State& state, size_t reiterations, size_t nodeId, size_t commonSeed) = 0;
};
