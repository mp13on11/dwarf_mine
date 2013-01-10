#pragma once

#include <MonteCarloElf.h>


class SMPMonteCarloElf : public MonteCarloElf
{
public:
    virtual OthelloMove calculateMove(const OthelloState& left);
};
