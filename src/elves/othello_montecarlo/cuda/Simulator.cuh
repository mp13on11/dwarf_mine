/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

#pragma once

#include "Random.cuh"
#include "State.cuh"
#include "Debug.cuh"

class CudaSimulator
{
private:
    size_t _playfieldIndex;
    size_t _playfieldX;
    size_t _playfieldY;
    CudaGameState* _state;
    curandState* _deviceState;
    size_t _randomSeed;
    float* _randomValues;

public:
    __device__ CudaSimulator(CudaGameState* state, float* randomValues, size_t randomSeed)
        : _playfieldIndex(threadIdx.x), _playfieldX(_playfieldIndex % FIELD_DIMENSION), _playfieldY(_playfieldIndex / FIELD_DIMENSION),
            _state(state), _randomSeed(randomSeed), _randomValues(randomValues)
    {
    }

    __device__ bool isMaster()
    {
        return _playfieldIndex == 0;
    }
    
    __device__ void calculatePossibleMoves()
    {
        __syncthreads();
        
        _state->possible[threadIdx.x] = false;
    
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
        bool foundEnemy = false;
        Player enemyPlayer = _state->getEnemyPlayer();
        int neighbourX = _playfieldX + directionX;
        int neighbourY = _playfieldY + directionY;
        
        int neighbourIndex = neighbourY * FIELD_DIMENSION + neighbourX;
        while (_state->inBounds(neighbourX, neighbourY) && _state->field[neighbourIndex] == enemyPlayer)
        {
            foundEnemy = true;
            
            neighbourX += directionX;
            neighbourY += directionY;
            neighbourIndex = neighbourY * FIELD_DIMENSION + neighbourX;
        }

        if (_state->inBounds(neighbourX, neighbourY) && _state->field[neighbourIndex] == _state->currentPlayer)
        {
            if (foundEnemy)
                _state->possible[_playfieldIndex] |= foundEnemy;
        }
    }

    __device__ size_t countPossibleMoves()
    {
        return numberOfMarkedFields(_state->possible);
    }

    // this function may deliver different results for the threads, so it should be only called once per block
    __device__ size_t getRandomMoveIndex(size_t moveCount, float fakedRandom = -1)
    {
        __shared__ size_t randomMoveCounter;
        if (_playfieldIndex == 0)
        {
            if (moveCount > 1)
            {
				randomMoveCounter = randomNumber(_randomValues, &_randomSeed, moveCount, fakedRandom); 
            }
            else
            {
                randomMoveCounter = 0;
            }
        }
        __syncthreads();
        cassert(randomMoveCounter < moveCount, "Detected invalid moveCounter %lu - maximum %lu\n", randomMoveCounter, moveCount);

        size_t possibleMoveCounter = 0;
        for (size_t i = 0; i < _state->size; ++i)
        {
            if (_state->possible[i])
            {
                if (possibleMoveCounter == randomMoveCounter)
                {
                    return i;
                }
                possibleMoveCounter++;
            }
        }
        cassert(false, "Could not find array index for move #%d - found only %lu possible moves in field of size %lu\n", randomMoveCounter, possibleMoveCounter, _state->size);
        return 96;
    }

    __device__ bool canFlipInDirection(size_t moveIndex, int directionX, int directionY)
    {
        Player enemyPlayer = _state->getEnemyPlayer();

        for (size_t currentIndex = _playfieldIndex; _state->inBounds(currentIndex); currentIndex += directionY * _state->sideLength + directionX)
        {
            if(_state->field[currentIndex] != enemyPlayer)
            {
                return (_state->field[currentIndex] == _state->currentPlayer && currentIndex != _playfieldIndex);
            }
        }
        return false;
    }

    __device__ void flipInDirection(size_t moveIndex, int directionX, int directionY, size_t limit)
    {
        Player enemyPlayer = _state->getEnemyPlayer();

        for (size_t currentIndex = _playfieldIndex; _state->inBounds(currentIndex); currentIndex += directionY * _state->sideLength + directionX)
        {
            if(_state->oldField[currentIndex] == enemyPlayer)
            {
                _state->field[currentIndex] = _state->currentPlayer;
            }
            else
            {
                break;
            }
        }
    }

    __device__ void flipEnemyCounter(size_t moveIndex, size_t limit)
    {
        __syncthreads();

        _state->oldField[_playfieldIndex] = _state->field[_playfieldIndex];

        __syncthreads();

        int directionX = _playfieldX - moveIndex % _state->sideLength;
        int directionY = _playfieldY - moveIndex / _state->sideLength;

        bool flip = false;
        if (abs(directionX) <= 1 && abs(directionY) <= 1 && moveIndex != _playfieldIndex)
        {
            flip = canFlipInDirection(moveIndex, directionX, directionY);
        }

        __syncthreads();

        if (flip)
        {
            flipInDirection(moveIndex, directionX, directionY, limit);
        }

        __syncthreads();

        if (_playfieldIndex == moveIndex && threadIdx.x == moveIndex)
        {
            _state->field[_playfieldIndex] = _state->currentPlayer;
        }
    }
};