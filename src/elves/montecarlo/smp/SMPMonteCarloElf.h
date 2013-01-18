#pragma once

#include <MonteCarloElf.h>


class SMPMonteCarloElf : public MonteCarloElf
{
public:
    virtual OthelloResult calculateBestMove(const OthelloState& state);
};
