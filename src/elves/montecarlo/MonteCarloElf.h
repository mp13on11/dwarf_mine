#pragma once

#include <Elf.h>
#include <OthelloMove.h>
#include <OthelloState.h>

class MonteCarloElf : public Elf
{
public:
    virtual OthelloMove calculateMove(const OthelloState& left) = 0;
    virtual void run(std::istream& input, std::ostream& output);
};
