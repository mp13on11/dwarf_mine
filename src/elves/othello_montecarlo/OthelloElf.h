#pragma once

#include <Elf.h>
#include <State.h>
#include <Result.h>
#include <vector>

class OthelloElf : public Elf
{
public:
    virtual std::vector<Result> getMovesFor(State& state, size_t reiterations, size_t nodeId, size_t commonSeed) = 0;
};
