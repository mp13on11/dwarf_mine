#pragma once

#include <vector>
#include <random>
#include <memory>
#include <functional>
#include "OthelloState.h"
#include "OthelloMove.h"
#include "OthelloUtil.h"

class OthelloNode;

typedef std::vector<OthelloNode> NodeList;

class OthelloNode
{

public:
    OthelloNode(const OthelloNode& node);
    OthelloNode(const OthelloState& state);
    ~OthelloNode();
    OthelloNode& operator=(const OthelloNode& node);

    bool hasUntriedMoves() const;
    bool hasChildren() const;
    OthelloNode& getFavoriteChild();
    OthelloNode& getRandomChildNode(RandomGenerator generator);
    OthelloNode& addChild(OthelloMove& move, const OthelloState& state);
    const NodeList& getChildren() const;
    OthelloMove getTriggerMove() const;
    OthelloMove getRandomMove(RandomGenerator generator) const;
    OthelloMove getRandomUntriedMove(RandomGenerator generator) const;
    void removeFromUntriedMoves(const OthelloMove& move);
    void updateSuccessProbability(bool hasWon);
    Player currentPlayer() const;
    OthelloNode& parent() const;
    bool hasParent() const;
    double successRate() const;
    OthelloResult collectedResult() const;

private:

    MoveList _untriedMoves;
    OthelloNode* _parent;
    NodeList _children;
    std::shared_ptr<OthelloMove> _triggerMove;
    std::shared_ptr<OthelloState> _state;
    size_t _visits;
    size_t _wins;
    void setParent(OthelloNode& parent);
    void setTriggerMove(OthelloMove& move);
};

inline const NodeList& OthelloNode::getChildren() const
{
    return _children;
}