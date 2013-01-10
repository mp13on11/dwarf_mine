#pragma once

#include <MatrixElf.h>

class SMPMonteCarloElf : public MatrixElf
{
public:
    virtual OthelloMove calculateMove(const OthelloState& left);
};
