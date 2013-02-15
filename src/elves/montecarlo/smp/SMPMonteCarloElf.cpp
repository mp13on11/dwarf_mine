#include "SMPMonteCarloElf.h"
#include "OthelloNode.h"
#include "OthelloUtil.h"
#include <functional>
#include <random>
#include <iostream>
#include <omp.h>
#include <chrono>
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
    }   
}

OthelloNode* SMPMonteCarloElf::select(OthelloNode* node, OthelloState& state, RandomGenerator generator)
{
    while(!node->hasUntriedMoves() && node->hasChildren())
    {
        node = &(node->getRandomChildNode(generator));
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
    do
    {
        node->updateSuccessProbability(state.hasWon(player));
        node = &(node->parent());
    }
    while(node->hasParent());
}

OthelloResult SMPMonteCarloElf::getBestMoveFor(OthelloState& rootState, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    size_t threadCount = omp_get_max_threads();

    for (size_t i = 0; i < threadCount; ++i)
    {
        mt19937 engine(OthelloHelper::generateUniqueSeed(nodeId, i, commonSeed));
        _generators.push_back([&engine](size_t limit){ 
            uniform_int_distribution<size_t> distribution(0, limit - 1);
            return distribution(engine);
        });    
    }
    

    OthelloNode rootNode(rootState);

    expand(rootState, rootNode);
    
    #pragma omp parallel for shared(rootState, rootNode) 
    for (size_t i = 0; i < reiterations; ++i)
    {

        OthelloNode* currentNode = &rootNode;
        OthelloState currentState = rootState;

        currentNode = select(currentNode, currentState, _generators[omp_get_thread_num()]);
        //expand() moved before parallelization
        rollout(currentState, _generators[omp_get_thread_num()]);
        backPropagate(currentNode, currentState, rootNode.currentPlayer());
    }
    
    auto result = rootNode.getFavoriteChild().collectedResult();
    result.iterations = reiterations;
    return result;
}