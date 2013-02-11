#pragma once

#include <Elf.h>
#include <OthelloUtil.h>
#include <OthelloState.h>

class MonteCarloElf : public Elf
{
public:
    virtual OthelloResult getBestMoveFor(OthelloState& state, size_t reiterations) = 0;
};
