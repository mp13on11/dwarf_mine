#pragma once

#include <Elf.h>

template<typename T>

class MonteCarloElf : public Elf
{
public:

    virtual OthelloMove calculateMove(const OthelloState& left) = 0;
    virtual void run(std::istream& input, std::ostream& output);
};
