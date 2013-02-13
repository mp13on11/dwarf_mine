#pragma once

#include <Elf.h>
#include <OthelloState.h>
#include <OthelloResult.h>

class MonteCarloElf : public Elf
{
public:
    virtual OthelloResult getBestMoveFor(OthelloState& state, size_t reiterations) = 0;
};
