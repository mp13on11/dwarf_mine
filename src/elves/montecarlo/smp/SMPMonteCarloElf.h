#pragma once

#include <MonteCarloElf.h>
#include "OthelloUtil.h"
#include "OthelloNode.h"

class SMPMonteCarloElf : public MonteCarloElf
{
public:
    virtual OthelloResult getBestMoveFor(OthelloState& state, size_t reiterations);
private:
    RandomGenerator _generator;

    void expand(OthelloNode* node, OthelloState& state);
    OthelloNode* select(OthelloNode* node, OthelloState& state);
    void rollout(OthelloState& state);
    void backPropagate(OthelloNode* node, OthelloState& state);
};
