#pragma once

#include <Elf.h>
#include <OthelloState.h>
#include <OthelloResult.h>
#include <vector>

class MonteCarloElf : public Elf
{
public:
    virtual std::vector<OthelloResult> getMovesFor(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed) = 0;
};
