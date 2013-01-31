#pragma once

typedef struct _Move
{
	size_t x;
	size_t y;
} Move;

typedef struct _MoveVector
{
	size_t length;
	Move* data;

	__device__ _MoveVector(size_t length)
	{
		this->length = length;
		this->data = new Move[length];
	}

	__device__ ~_MoveVector()
	{
		delete[] this->data;
	}
} MoveVector;
