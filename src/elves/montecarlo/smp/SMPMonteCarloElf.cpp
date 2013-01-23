#include "SMPMonteCarloElf.h"
#include "OthelloNode.h"
#include "OthelloUtil.h"
#include <functional>
#include <random>
#include <iostream>

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

void SMPMonteCarloElf::backPropagate(OthelloNode* node, OthelloState& state)
{
    do
    {
        node->updateSuccessProbability(state.hasWon(node->currentPlayer()));
        node = &(node->parent());
    }
    while(node->hasParent());
}

OthelloResult SMPMonteCarloElf::getBestMoveFor(OthelloState& state, size_t reiterations)
{

    std::mt19937 engine(time(nullptr));
    _generator = [&engine](size_t limit){ 
        uniform_int_distribution<size_t> distribution(0, limit - 1);
        return distribution(engine);
    };

    OthelloNode node(state);

    while (node.hasUntriedMoves())
    {
        auto childState = state;
        expand(&node, childState);
    }

    for(size_t i = 0; i < reiterations; ++i)
    {
        OthelloNode* currentNode = &node;
        OthelloState currentState = state;

        currentNode = select(currentNode, currentState);
        //expand() moved before parallelization
        rollout(currentState);
        backPropagate(currentNode, currentState);
    }

    auto result = node.getFavoriteChild().collectedResult();
    return result;
}