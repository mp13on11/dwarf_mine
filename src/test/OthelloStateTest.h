#pragma once

#include <gtest/gtest.h>
#include <othello_montecarlo/State.h>

class OthelloStateTest : public testing::Test
{
public:
    virtual void SetUp();
    virtual void TearDown();

protected:
    State* state;
};
