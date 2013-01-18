#pragma once

#include <vector>
#include <random>
#include <memory>
#include <functional>
#include "OthelloState.h"
#include "OthelloMove.h"
#include "OthelloResult.h"

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
    OthelloNode& getRandomChildNode();
    OthelloMove& getTriggerMove();
    OthelloMove& getRandomMove();
    OthelloMove& getRandomUntriedMove();
    OthelloNode& addChild(OthelloMove& move, const OthelloState& state);
    void updateSuccessProbability(bool hasWon);
    Player currentPlayer();
    OthelloNode& parent();
    bool hasParent();
    double successRate();
    OthelloResult collectedResult();

private:
    static std::mt19937 _randomGenerator;

    std::vector<OthelloMove> _untriedMoves;
    OthelloNode* _parent;
    std::vector<OthelloNode> _children;
    std::shared_ptr<OthelloMove> _triggerMove;
    std::shared_ptr<OthelloState> _state;
    size_t _visits;
    size_t _wins;
    void setParent(OthelloNode& parent);
    void setTriggerMove(OthelloMove& move);

protected:
    virtual std::function<size_t()> getGenerator(size_t min, size_t max);
};
