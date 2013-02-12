#include "OthelloNode.h"

#include <random>
#include <stdexcept>
#include <cassert>

using namespace std;

OthelloNode::OthelloNode(const OthelloNode& node)
{
    _untriedMoves = node._untriedMoves;
    _state = node._state;
    _parent = node._parent;
    _children = node._children;
    //_visits = node._visits;
    //_wins = node._wins;
    _visits = node._visits.load();
    _wins = node._wins.load();
    _triggerMove = node._triggerMove;
}

OthelloNode::OthelloNode(const OthelloState& state) 
    : _parent(nullptr), _triggerMove(nullptr), _state(make_shared<OthelloState>(state)), _visits(0), _wins(0)
{
    _untriedMoves = _state->getPossibleMoves();
}

OthelloNode::~OthelloNode()
{

}

OthelloNode& OthelloNode::operator=(const OthelloNode& node)
{
    _untriedMoves = node._untriedMoves;
    _state = node._state;
    _parent = node._parent;
    _children = node._children;
    // _visits = node._visits;
    // _wins = node._wins;
    _visits = node._visits.load();
    _wins = node._wins.load();
    _triggerMove = node._triggerMove;
    return *this;
}

bool OthelloNode::hasChildren() const
{
    return !_children.empty();
}

OthelloNode& OthelloNode::getRandomChildNode(RandomGenerator generator)
{
    if (!hasChildren())
    {
        throw runtime_error("OthelloNode::getRandomChildNode(): No child");
    }
    if (_children.size() == 1)
    {
        return _children[0];
    }
    return _children[generator(_children.size())];
}

OthelloMove OthelloNode::getTriggerMove() const
{
    if (!_triggerMove)
    {
        throw runtime_error("OthelloNode::getTriggerMove(): No trigger move");
    }
    return *(_triggerMove.get());
}

void OthelloNode::setTriggerMove(OthelloMove& move)
{   
    _triggerMove = make_shared<OthelloMove>(move);
}

OthelloMove OthelloNode::getRandomMove(RandomGenerator generator) const
{
    auto moves = _state->getPossibleMoves();
    return moves[generator(moves.size())];
}

bool OthelloNode::hasUntriedMoves() const
{
    return !_untriedMoves.empty();
}

void OthelloNode::removeFromUntriedMoves(const OthelloMove& move)
{
    auto i = _untriedMoves.begin();
    for (; i != _untriedMoves.end(); ++i)
    {
        if (*i == move)
            break;
    }
    if (i != _untriedMoves.end())
        _untriedMoves.erase(i);
}

void OthelloNode::setParent(OthelloNode& node)
{
    _parent = &node;
}

OthelloMove OthelloNode::getRandomUntriedMove(RandomGenerator generator) const
{
    if (!hasUntriedMoves())
    {
        throw runtime_error("OthelloNode::getRandomUntriedMove(): No untried move");
    }
    return _untriedMoves[generator(_untriedMoves.size())];
}

OthelloNode& OthelloNode::addChild(OthelloMove& move, const OthelloState& state)
{
    _children.push_back(state);
    _children.back()._parent = this;
    _children.back().setTriggerMove(move);
    return _children.back();
}

void OthelloNode::updateSuccessProbability(bool hasWon)
{
    ++_visits;
    if(hasWon)
        ++_wins;
}
    
Player OthelloNode::currentPlayer() const
{
    return _state->getCurrentPlayer();
}
    
OthelloNode& OthelloNode::parent() const
{
    return *_parent;
}
    
bool OthelloNode::hasParent() const
{
    return _parent != nullptr;
}

double OthelloNode::successRate() const
{
    if (_visits == 0)
        return 0;
    return 1.0 * _wins / _visits;
}

OthelloResult OthelloNode::collectedResult() const
{
    auto move = getTriggerMove();
    return OthelloResult{(size_t)move.x, (size_t)move.y, _visits, _wins, 0};
}

OthelloNode& OthelloNode::getFavoriteChild()
{
    assert(_children.size() > 0);
    size_t favorite = 0;
    if (_children.size() > 1)
    {
        for (size_t i = 0; i < _children.size(); ++i)
        {
            if (_children[favorite].successRate() < _children[i].successRate())
            {
                favorite = i;
            }
        }
    }
    return _children[favorite];
}
