#pragma once

#include <elves/quadratic_sieve/QuadraticSieveElf.h>
#include <elves/common-factorization/BigInt.h>

#include <memory>
#include <utility>
#include <gtest/gtest.h>

class FactorizationTest : public testing::TestWithParam<std::pair<BigInt, BigInt>>
{
protected:
    virtual void SetUp();

    BigInt p;
    BigInt q;
    BigInt product;
};
