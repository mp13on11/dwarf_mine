#pragma once

#include <MonteCarloElf.h>
#include "OthelloUtil.h"
#include "OthelloNode.h"
#include <chrono>
#include <vector>

class SMPMonteCarloElf : public MonteCarloElf
{
public:
    virtual OthelloResult getBestMoveFor(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed);
private:
    std::vector<RandomGenerator> _generators;
    std::chrono::high_resolution_clock::time_point _end;

    void expand(OthelloState& state, OthelloNode& node);
    OthelloNode* select(OthelloNode* node, OthelloState& state, RandomGenerator generator);
    void rollout(OthelloState& state, RandomGenerator generator);
    void backPropagate(OthelloNode* node, OthelloState& state, Player player);
};

