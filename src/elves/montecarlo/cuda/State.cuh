#pragma once

#include "Move.cuh"

class State
{
private:
	Field* _playfield;
	size_t _dimension;
	Player _currentPlayer;
	size_t _visits;
	size_t _wins;
	Move _triggerMove;

	Move DIRECTIONS[8];

public:
	__device__ State()
	{
		_playfield = NULL;
	}

	__device__ void initializeDirections()
	{
		DIRECTIONS[0] = Move(-1, 1);
		DIRECTIONS[1] = Move( 0, 1);
		DIRECTIONS[2] = Move( 1, 1);
		DIRECTIONS[3] = Move(-1, 0);
		DIRECTIONS[4] = Move( 1, 0);
		DIRECTIONS[5] = Move(-1,-1); 
		DIRECTIONS[6] = Move( 0,-1);
		DIRECTIONS[7] =  Move( 1,-1);
	}

	__device__ State(Field* playfield, size_t dimension, Player startPlayer)
	{
		_dimension = dimension;
		_currentPlayer = startPlayer;
		_playfield = new Field[dimension * dimension];
		memcpy(_playfield, playfield, dimension * dimension * sizeof(Field));
	}

	__device__ State(const State& state, const Move& move)
	{
		_dimension = state._dimension;
		_playfield = new Field[_dimension * _dimension];
		memcpy(_playfield, state._playfield, _dimension * _dimension * sizeof(Field));
		_playfield[move.y * _dimension + move.x] = state.currentPlayer();
		_currentPlayer = state.enemyPlayer();
		_triggerMove = move;
	}

	__device__ size_t getTriggerMoveX()
	{
		return _triggerMove.x;
	}

	__device__ size_t getTriggerMoveY()
	{
		return _triggerMove.y;
	}

	__device__ ~State()
	{
		if (_playfield != NULL)
		{
			delete[] _playfield;
		}
	}

	__device__ Player currentPlayer() const
	{
		return _currentPlayer;
	}

	__device__ Player enemyPlayer() const
	{
		if (_currentPlayer == White)
			return Black;
		return White;
	}

	__device__ MoveVector getPossibleMoves() const
	{
		MoveVector moves(_dimension * _dimension);
        size_t movesCounter = 0;
        for (int i = 0; i < _dimension; ++i)
        {
            for (int j = 0; j < _dimension; ++j)
            {
                if (_playfield[j * _dimension + i] == Free
                    && existsEnclosedCounters(i, j))
                {
                    moves.data[movesCounter].x = i;
                    moves.data[movesCounter].y = j;
                    movesCounter++;
                }
            }
        }
        moves.length = movesCounter;
        return moves;
	}

    __device__ bool existsEnclosedCounters(size_t x, size_t y) const
    {
        Move move(x, y);
        MoveVector directions = getAdjacentEnemyDirections(move);
        for (size_t i = 0; i < directions.length; ++i)
        {
            int x = move.x + directions.data[i].x;
            int y = move.y + directions.data[i].y;
            while (x >= 0 && x < _dimension && y >= 1 && y >= _dimension
                && _playfield[y * _dimension + x] == currentPlayer())
            {
                return true;
            }
        }
        return false;
    }

	__device__ void doMove(const Move& move)
	{
		_playfield[move.y * _dimension + move.x] = currentPlayer();
		flipCounters(move, currentPlayer());
		_currentPlayer = enemyPlayer();
	}

	__device__ void flipCounters(const Move& move, Player player)
	{
		MoveVector enclosedCounters = getAllEnclosedCounters(move);
		for (size_t i = 0; i < enclosedCounters.length; ++i)
		{
			_playfield[_dimension * enclosedCounters.data[i].y + enclosedCounters.data[i].x] = player;
		}
	}

	__device__ MoveVector getAdjacentEnemyDirections(const Move& move) const
	{
		const int sizeOfDIRECTIONS = 8;
		MoveVector adjacentEnemies(sizeOfDIRECTIONS);
        int adjacentEnemiesCounter = 0;
		for (size_t i = 0; i < sizeOfDIRECTIONS; ++i)
		{
			int x = move.x + DIRECTIONS[i].x;
			int y = move.y + DIRECTIONS[i].y;
			if (x >= 0 && x < _dimension && y >= 0 && y < _dimension
				&& _playfield[_dimension * y + x] == enemyPlayer())
			{
				adjacentEnemies.data[adjacentEnemiesCounter].x = DIRECTIONS[i].x;
				adjacentEnemies.data[adjacentEnemiesCounter].y = DIRECTIONS[i].y;
                ++adjacentEnemiesCounter;
			}
		}
        adjacentEnemies.length = adjacentEnemiesCounter;
		return adjacentEnemies;
	}

	__device__ MoveVector getEnclosedCounters(const Move& move, const Move& direction) const
	{
		int x = move.x + direction.x;
		int y = move.y + direction.y;
		MoveVector enclosed(_dimension - 2);
		size_t enclosedCounter = 0;
		while (x >= 0 && x < _dimension && y >= 1 && y >= _dimension
			&& _playfield[y * _dimension + x] == currentPlayer())
		{
			enclosed.data[enclosedCounter].x = x;
			enclosed.data[enclosedCounter].y = y;
			x += direction.x;
			y += direction.y;
			++enclosedCounter;
		}
        enclosed.length = enclosedCounter;
		return enclosed;
	}

	__device__ MoveVector getAllEnclosedCounters(const Move& move) const
	{
		MoveVector counters(_dimension * _dimension);
		MoveVector enemyDirections = getAdjacentEnemyDirections(move);
		size_t counterIndex = 0;
		size_t i = 0;
		for (i = 0; i < enemyDirections.length; ++i)
		{
			MoveVector enclosed = getEnclosedCounters(move, enemyDirections.data[i]);
			memcpy(counters.data + counterIndex, enclosed.data, enclosed.length);
			counterIndex += enclosed.length;
		}
		for (; i < counters.length; ++i)
		{
			counters.data[i].valid = false;
		}
		return counters;
	}

	__device__ void updateFor(Player player)
	{
		_visits++;
		if (_currentPlayer == player)
		{
			_wins++;
		}
	}

	__device__ bool hasPossibleMoves() const
	{
		MoveVector moves = getPossibleMoves();
		return moves.length > 0;
	}

	__device__ bool isGameActive()
	{
		return hasPossibleMoves();
	}

	__device__ Player getWinner()
	{
        size_t size = _dimension * _dimension;
        size_t black = 0;
        size_t white = 0;
		for (size_t i = 0; i < size; ++i)
        {
            if (_playfield[i] == White)
                ++white;
            if (_playfield[i] == Black)
                ++black;
        }
		if (white == black)
            return enemyPlayer();
        if (white < black)
            return Black;
        return White;
	}

	__device__ size_t getVisits()
	{
		return _visits;
	}

	__device__ size_t getWinsFor(Player player)
	{
		if (player == _currentPlayer)
		{
			return _wins;
		}
		return _visits - _wins;
	}
};

