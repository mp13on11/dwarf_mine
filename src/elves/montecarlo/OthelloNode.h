#pragma once

#include <vector>
#include <random>
#include <memory>
#include <functional>
#include "OthelloState.h"
#include "OthelloMove.h"
#include "OthelloUtil.h"

class OthelloNode
{

public:
    OthelloNode(const OthelloNode& node);
    OthelloNode(const OthelloState& state);
    ~OthelloNode();
    OthelloNode& operator=(const OthelloNode& node);

    bool hasUntriedMoves();
    bool hasChildren();
    OthelloNode& getFavoriteChild();
    OthelloNode& getRandomChildNode(RandomGenerator generator);
    OthelloNode& addChild(OthelloMove& move, const OthelloState& state);
    std::vector<OthelloNode>& getChildren();
    OthelloMove getTriggerMove();
    OthelloMove getRandomMove(RandomGenerator generator);
    OthelloMove getRandomUntriedMove(RandomGenerator generator);
    void removeFromUntriedMoves(const OthelloMove& move);
    void updateSuccessProbability(bool hasWon);
    Player currentPlayer();
    OthelloNode& parent();
    bool hasParent();
    double successRate();
    OthelloResult collectedResult();

private:

    std::vector<OthelloMove> _untriedMoves;
    OthelloNode* _parent;
    std::vector<OthelloNode> _children;
    std::shared_ptr<OthelloMove> _triggerMove;
    std::shared_ptr<OthelloState> _state;
    size_t _visits;
    size_t _wins;
    void setParent(OthelloNode& parent);
    void setTriggerMove(OthelloMove& move);
};

inline std::vector<OthelloNode>& OthelloNode::getChildren()
{
    return _children;
}