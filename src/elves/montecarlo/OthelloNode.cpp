#include "OthelloNode.h"

#include <random>
#include <stdexcept>
#include <cassert>

using namespace std;

std::mt19937 OthelloNode::_randomGenerator;

function<size_t()> OthelloNode::getGenerator(size_t min, size_t max)
{
	uniform_int_distribution<size_t> distribution(min, max - 1);
	return bind(distribution, OthelloNode::_randomGenerator);
}

OthelloNode::OthelloNode(const OthelloNode& node)
{
	_untriedMoves = node._untriedMoves;
	_state = node._state;
	_parent = node._parent;
	_children = node._children;
	_visits = node._visits;
	_wins = node._wins;
	_triggerMove = node._triggerMove;
}

OthelloNode::OthelloNode(const OthelloState& state) 
	: _parent(nullptr), _triggerMove(nullptr), _state(make_shared<OthelloState>(state))
{
	_untriedMoves = _state->getPossibleMoves();
}

OthelloNode& OthelloNode::operator=(const OthelloNode& node)
{
	_untriedMoves = node._untriedMoves;
	_state = node._state;
	_parent = node._parent;
	_children = node._children;
	_visits = node._visits;
	_wins = node._wins;
	_triggerMove = node._triggerMove;
	return *this;
}

bool OthelloNode::hasChildren()
{
	return !_children.empty();
}

OthelloNode& OthelloNode::getRandomChildNode()
{
	if (!hasChildren())
	{
		throw runtime_error("OthelloNode::getRandomChildNode(): No child");
	}
	if (_children.size() == 1)
	{
		return _children[0];
	}
	auto random = getGenerator(0, _children.size());
	return _children[random()];
}

OthelloMove& OthelloNode::getTriggerMove()
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

OthelloMove& OthelloNode::getRandomMove()
{
	auto moves = _state->getPossibleMoves();
	auto random = getGenerator(0, moves.size());
	return moves[random()];
}

bool OthelloNode::hasUntriedMoves()
{
	return !_untriedMoves.empty();
}

OthelloMove& OthelloNode::getRandomUntriedMove()
{
	if (!hasUntriedMoves())
	{
		throw runtime_error("OthelloNode::getRandomUntriedMove(): No untried move");
	}
	auto random = getGenerator(0, _untriedMoves.size());
	return _untriedMoves[random()];
}

OthelloNode& OthelloNode::addChild(OthelloMove& move, const OthelloState& state)
{
	OthelloNode child(state);
	_children.emplace_back(state);
	_children.back().setTriggerMove(move);
	return _children.back();
}

void OthelloNode::updateSuccessProbability(bool hasWon)
{
	++_visits;
	if(hasWon)
		++_wins;
}
	
Player OthelloNode::currentPlayer()
{
	return _state->getCurrentPlayer();
}
	
OthelloNode& OthelloNode::parent()
{
	return *(_parent.get());
}
	
bool OthelloNode::hasParent()
{
	return _parent != nullptr;
}

double OthelloNode::successRate()
{
	return 1.0 * _wins / _visits;
}

OthelloNode& OthelloNode::getFavoriteChild()
{
	assert(_children.size() > 1);
	size_t favorite = 0;
	for (size_t i = 1; i < _children.size(); ++i)
	{
		if (_children[favorite].successRate() < _children[i].successRate())
		{
			favorite = i;
		}
	}
	return _children[favorite];
}
