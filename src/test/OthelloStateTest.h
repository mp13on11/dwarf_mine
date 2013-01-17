#pragma once

#include <gtest/gtest.h>
#include <montecarlo/OthelloState.h>

class OthelloStateTest : public testing::Test
{
public:
    virtual void SetUp();
    virtual void TearDown();

protected:
    OthelloState* state;
};
