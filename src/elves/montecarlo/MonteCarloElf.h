#pragma once

#include <Elf.h>
#include <OthelloResult.h>
#include <OthelloState.h>

class MonteCarloElf : public Elf
{
public:
    virtual OthelloResult calculateBestMove(const OthelloState& state) = 0;
    virtual void run(std::istream& input, std::ostream& output);
};
