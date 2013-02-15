#include "SMPMonteCarloElf.h"
#include "OthelloNode.h"
#include "OthelloUtil.h"
#include <functional>
#include <random>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <cassert>
#include <ratio>

using namespace std;

void SMPMonteCarloElf::expand(OthelloState& state, OthelloNode& node)
{
    auto moves = state.getPossibleMoves();
    for (const auto& move : moves)
    {
        auto childState = state;
        childState.doMove(move);
        node.addChild(move, childState);
        node.removeFromUntriedMoves(move);
    }   
}

OthelloNode* SMPMonteCarloElf::select(OthelloNode* node, OthelloState& state, RandomGenerator generator)
{
    while(!node->hasUntriedMoves() && node->hasChildren())
    {
        node = node->getRandomChildNode(generator);
        OthelloMove move = node->getTriggerMove();
        state.doMove(move);
    }
    return node;
}

void SMPMonteCarloElf::rollout(OthelloState& state, RandomGenerator generator)
{
    while(state.hasPossibleMoves())
    {
        OthelloMove move = state.getRandomMove(generator);
        state.doMove(move);
    }
}

void SMPMonteCarloElf::backPropagate(OthelloNode* node, OthelloState& state, Player player)
{
    node->updateSuccessProbability(state.hasWon(player));
}

OthelloResult SMPMonteCarloElf::getBestMoveFor(OthelloState& rootState, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    size_t threadCount = omp_get_max_threads();
    vector<mt19937> engines;
    for (size_t i = 0; i < threadCount; ++i)
    {
        engines.emplace_back(OthelloHelper::generateUniqueSeed(nodeId, i, commonSeed));
    }

    auto generator = [&engines](size_t limit){ 
        uniform_int_distribution<size_t> distribution(0, limit - 1);
        return distribution(engines[omp_get_thread_num()]);
    };    


    OthelloNode rootNode(rootState);

    expand(rootState, rootNode);

    auto childNodes = rootNode.getChildren();
    
    #pragma omp parallel for shared(rootState, rootNode) 
    for (size_t i = 0; i < reiterations; ++i)
    {
        OthelloState currentState = rootState;

        OthelloNode* currentNode = select(&rootNode, currentState, generator);

        //expand() moved before parallelization
          
        rollout(currentState, generator);
        backPropagate(currentNode, currentState, rootNode.currentPlayer());
    }

    auto result = rootNode.getFavoriteChild()->collectedResult();
    result.iterations = reiterations;
    return result;
}