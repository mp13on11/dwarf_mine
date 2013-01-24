#pragma once

#include <MonteCarloElf.h>
#include "OthelloUtil.h"
#include "OthelloNode.h"
#include <chrono>

class SMPMonteCarloElf : public MonteCarloElf
{
public:
    virtual OthelloResult getBestMoveFor(OthelloState& state, size_t reiterations);
private:
    RandomGenerator _generator;
    std::chrono::high_resolution_clock::time_point _end;

    void expand(OthelloNode* node, OthelloState& state);
    OthelloNode* select(OthelloNode* node, OthelloState& state);
    void rollout(OthelloState& state);
    void backPropagate(OthelloNode* node, OthelloState& state, Player player);

    void startTimer(size_t runtime_in_seconds);
    bool allowedToRun();
};

