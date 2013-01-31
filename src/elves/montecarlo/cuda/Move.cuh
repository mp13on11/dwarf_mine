#pragma once

typedef struct _Move
{
	int x;
	int y;
	bool valid; 
	
	__device__ _Move()
	{
		valid = true;
	}

	__device__ _Move(int _x, int _y)
	{
		x = _x;
		y = _y;
	}
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
