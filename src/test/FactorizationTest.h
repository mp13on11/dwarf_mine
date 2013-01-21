#pragma once

#include <elves/factorize/FactorizationElf.h>
#include <factorize/BigInt.h>

#include <memory>
#include <map>
#include <gtest/gtest.h>

class FactorizationTest : public testing::TestWithParam<std::pair<BigInt, BigInt>>
{
protected:
    virtual void SetUp();

    BigInt p;
    BigInt q;
    BigInt product;
};
