#pragma once

#include "CudaUtil.cuh"

class CudaSimulator
{
private:
    size_t _playfieldIndex;
    size_t _playfieldX;
    size_t _playfieldY;
    CudaGameState* _state;
    curandState* _deviceState;

public:
    __device__ CudaSimulator(CudaGameState* state, curandState* deviceState)
        : _playfieldIndex(threadIdx.x), _playfieldX(_playfieldIndex % FIELD_DIMENSION), _playfieldY(_playfieldIndex / FIELD_DIMENSION),
            _state(state), _deviceState(deviceState)
    {
    }

    __device__ bool isMaster()
    {
        return _playfieldIndex == 0;
    }
    
    __device__ void calculatePossibleMoves()
    {
        _state->possible[_playfieldIndex] = false;
    
        __syncthreads();

        if (_state->field[_playfieldIndex] == Free)
        {
            findPossibleMoves( 1,  1);
            findPossibleMoves( 1,  0);
            findPossibleMoves( 1, -1);
            findPossibleMoves( 0,  1);
            
            findPossibleMoves( 0, -1);
            findPossibleMoves(-1,  1);
            findPossibleMoves(-1,  0);
            findPossibleMoves(-1, -1);
        }

        __syncthreads();
    }

    __device__ void findPossibleMoves(int directionX, int directionY)
    {
        bool look = true;
        bool foundEnemy = false;
        Player enemyPlayer = _state->getEnemyPlayer();
        int neighbourX = _playfieldX + directionX;
        int neighbourY = _playfieldY + directionY;
        while (look)
        {
            int neighbourIndex = neighbourY * FIELD_DIMENSION + neighbourX;
            if (_state->inBounds(neighbourX, neighbourY))
            {
                if (_state->field[neighbourIndex] == Free)
                {
                    _state->possible[_playfieldIndex] |= false;
                    look = false;
                }
                else if(_state->field[neighbourIndex] == enemyPlayer)
                {
                    foundEnemy = true;
                }
                else if (_state->field[neighbourIndex] == _state->currentPlayer)
                {
                    _state->possible[_playfieldIndex] |= foundEnemy;
                    look = false;
                }
            }
            else
            {
                _state->possible[_playfieldIndex] |= false;
                look = false;
            }
            neighbourX += directionX;
            neighbourY += directionY;
        }
    }

    __device__ size_t countPossibleMoves()
    {
        // __syncthreads();
     //    __shared__ size_t moves[FIELD_DIMENSION * FIELD_DIMENSION];
     //    moves[playfieldIndex] = possibleMoves[playfieldIndex] ? 1 : 0;
        // return sum(moves, playfieldIndex, FIELD_DIMENSION * FIELD_DIMENSION);
        
        __syncthreads();
        size_t sum = 0;
        for (int i = 0; i < _state->size; i++)
        {
            if (_state->possible[i])
            {
                sum++;
                __syncthreads();
            }
        }
        return sum;
    }

    __device__ size_t getRandomMoveIndex(size_t moveCount, float fakedRandom = -1)
    {
        size_t randomMoveIndex = 0;
        if (moveCount > 1)
        {
            if (fakedRandom >= 0)
            {
                randomMoveIndex = fakedRandom * moveCount;
            }
            else
            {
                randomMoveIndex = randomNumber(_deviceState, moveCount);    
            }
        }
        size_t possibleMoveIndex = 0;
        for (size_t i = 0; i < _state->size; ++i)
        {
            if (_state->possible[i])
            {
                if (possibleMoveIndex == randomMoveIndex)
                {
                    return i;
                }
                possibleMoveIndex++;;
            }
        }
        return 0;
    }

    __device__ void flipDirection(size_t moveIndex, int directionX, int directionY)
    {
        int currentIndex = _playfieldIndex;
        Player enemyPlayer = _state->getEnemyPlayer();
        bool flip = false;

        for (currentIndex = _playfieldIndex; _state->inBounds(currentIndex); currentIndex += directionY * _state->sideLength + directionX)
        {
            if(_state->field[currentIndex] != enemyPlayer)
            {
                flip = (_state->field[currentIndex] == _state->currentPlayer && currentIndex != _playfieldIndex);
                break;
            }
        }
        __syncthreads();
        if (flip)
        {
            for (; currentIndex - moveIndex != 0 ; currentIndex -= directionY * _state->sideLength + directionX)
            {
                _state->field[currentIndex] = _state->currentPlayer;
            }
        }
    }

    __device__ void flipEnemyCounter(size_t moveIndex)
    {
        int directionX = _playfieldX - moveIndex % _state->sideLength;
        int directionY = _playfieldY - moveIndex / _state->sideLength;

        if (abs(directionX) <= 1 && abs(directionY) <= 1 && moveIndex != _playfieldIndex)
        {
            flipDirection(moveIndex, directionX, directionY);
        }
    }
};