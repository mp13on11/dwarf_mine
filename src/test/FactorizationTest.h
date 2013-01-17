#pragma once

#include <elves/factorize/FactorizationElf.h>

#include <memory>
#include <gtest/gtest.h>

class FactorizationTest : public testing::TestWithParam<const char*>
{
protected:

    virtual void SetUp();

    std::unique_ptr<FactorizationElf> elf;
};
