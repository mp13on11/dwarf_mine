#include "OthelloState.h"

#include <stdexcept>

using namespace std;

vector<OthelloMove> DIRECTIONS = {
	{ 1,1}, { 1,0}, { 1, -1},
	{ 0,1},         { 0, -1},
	{-1,1}, {-1,0}, {-1, -1}
};

OthelloState::OthelloState(const OthelloState& state)
	: _playfield(state._playfield), _sideLength(state._sideLength), _player(state._player)
{

}

OthelloState::OthelloState(size_t sideLength)
	: _sideLength((int)sideLength), _player(Player::White)
{
	if (_sideLength % 2 != 0)
	{
		throw logic_error("OthelloState::OthelloState: can not create playfield with odd size");
	}

	_playfield.resize(_sideLength * _sideLength);
	int center = _sideLength / 2 - 1;
	playfield({center    , center    }) = Field::White; // top left
	playfield({center    , center + 1}) = Field::Black; // top right
	playfield({center + 1, center    }) = Field::Black;
	playfield({center + 1, center + 1}) = Field::White;
}

void OthelloState::doMove(const OthelloMove& move)
{
	if (playfield(move) == Field::Free)
	{
		throw runtime_error("OthelloState::doMove(): Invalid move - place already allocated");
	}
	if (!onBoard(move))
	{
		throw runtime_error("OthelloState::doMove(): Move out of field bounds");
	}

	auto sandwichedCounterPositions = getAllSandwichCounters(move);

	_player = getCurrentEnemy();

	playfield(move) = _player;

	for (const auto& position : sandwichedCounterPositions)
	{
		playfield(position) = _player;
	}
}

bool OthelloState::onBoard(const OthelloMove& move) const
{
	return move.x < _sideLength && move.x >= 0 && move.y < _sideLength && move.y >= 0;
}

vector<OthelloMove> OthelloState::getAdjacentEnemyDirections(const OthelloMove& move)
{
	vector<OthelloMove> directions;
	for (const auto& direction : DIRECTIONS)
	{
		const auto position = move + direction;
		if (onBoard(position) && playfield(position) == getCurrentPlayer())
		{
			directions.push_back(direction);
		}
	}
	return directions;
}

vector<OthelloMove> OthelloState::getAllSandwichCounters(const OthelloMove& move)
{
	vector<OthelloMove> counters;
	vector<OthelloMove> enemyDirections = getAdjacentEnemyDirections(move);
	for (const auto& direction : enemyDirections)
	{
		vector<OthelloMove> enclosed = getSandwichedCounters(move, direction);
			for (const auto& e : enclosed)
			counters.push_back(e);
	}
	return counters;
}

vector<OthelloMove> OthelloState::getSandwichedCounters(const OthelloMove& move, const OthelloMove& direction)
{
	auto position = move + direction;
	vector<OthelloMove> counters;
	while (onBoard(position) && playfield(position) == getCurrentPlayer())
	{
		counters.push_back(position);
		position += direction;
	}
	if (onBoard(position) && playfield(position) == getCurrentEnemy())
	{
		return counters;
	}
	counters.clear();
	return counters;
}

vector<OthelloMove> OthelloState::getPossibleMoves()
{
	vector<OthelloMove> moves;
	for (int i = 0; i < _sideLength; ++i)
	{
		for (int j = 0; j < _sideLength; ++j)
		{
			auto position = OthelloMove{i, j};
			if (playfield(position) == Field::Free
				&& existsSandwichedCounters(position))
			{
				moves.push_back(move(position));
			}
		}
	}
	return moves;
}
// enclosed

bool OthelloState::existsSandwichedCounters(const OthelloMove& move)
{
	auto directions = getAdjacentEnemyDirections(move);
	for (const auto& direction : directions)
	{
		if (getSandwichedCounters(move, direction).size() > 0)
		{
			return true;
		}
	}
	return false;
}

// TODO optimize - cache of some fancy rule to detect a move
bool OthelloState::hasPossibleMoves()
{
	return getPossibleMoves().size() > 0;
}

bool OthelloState::hasWon(Player player)
{
	size_t playerCounter = 0;
	size_t enemyCounter = 0;
	for (int i = 0; i < _sideLength; ++i)
	{
		for (int j = 0; j < _sideLength; ++j)
		{
			OthelloMove position{i, j};
			Field field = playfield(position);
			if (field == Field::Free) 
				continue;
			if (field == player)
				playerCounter++;
			else 
				enemyCounter++;
		}
	}
	return playerCounter > enemyCounter;
}

Player OthelloState::getCurrentEnemy() const
{
	return _player == Field::Black ? Field::White : Field::Black;
}

Player OthelloState::getCurrentPlayer() const
{
	return _player;
}