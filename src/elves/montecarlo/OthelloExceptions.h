#pragma once

#include <stdexcept>
#include <string>
#include "OthelloMove.h"

class InvalidFieldSizeException : public std::logic_error
{
public:
    InvalidFieldSizeException(size_t sideLength);

private:
    static std::string constructMessage(size_t sideLength);
};

class InvalidMoveException : public std::runtime_error
{
public:
    InvalidMoveException(size_t sideLength, const OthelloMove& move);

private:
    static std::string constructMessage(size_t sideLength, const OthelloMove& move);
};

class OccupiedFieldException : public std::runtime_error
{
public:
    OccupiedFieldException(const OthelloMove& move);

private:
    static std::string constructMessage(const OthelloMove& move);
};