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

void SMPMonteCarloElf::expand(OthelloNode* node, OthelloState& state)
{
    OthelloMove move = node->getRandomUntriedMove(_generator);
    node->removeFromUntriedMoves(move);
    state.doMove(move);
    node->addChild(move, state);   
}

OthelloNode* SMPMonteCarloElf::select(OthelloNode* node, OthelloState& state)
{
    while(!node->hasUntriedMoves() && node->hasChildren())
    {
        node = &(node->getRandomChildNode(_generator));
        OthelloMove move = node->getTriggerMove();
        state.doMove(move);
    }
    return node;
}

void SMPMonteCarloElf::rollout(OthelloState& state)
{
    while(state.hasPossibleMoves())
    {
        OthelloMove move = state.getRandomMove(_generator);
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

OthelloResult SMPMonteCarloElf::getBestMoveFor(OthelloState& rootState, size_t reiterations)
{

    mt19937 engine(time(nullptr));
    _generator = [&engine](size_t limit){ 
        uniform_int_distribution<size_t> distribution(0, limit - 1);
        return distribution(engine);
    };

    OthelloNode rootNode(rootState);

    while (rootNode.hasUntriedMoves())
    {
        auto childState = rootState;
        expand(&rootNode, childState);
    }

 	#pragma omp parallel for shared(rootState, rootNode) 
	for (size_t i = 0; i < reiterations; ++i)
	{
        OthelloNode* currentNode = &rootNode;
        OthelloState currentState = rootState;

        currentNode = select(currentNode, currentState);
        //expand() moved before parallelization
        rollout(currentState);
        backPropagate(currentNode, currentState, rootNode.currentPlayer());
    }
    
    auto result = rootNode.getFavoriteChild().collectedResult();
    result.iterations = reiterations;
    return result;
}