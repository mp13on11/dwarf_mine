#pragma once

#include <gtest/gtest.h>
#include <matrix/MatrixSlicer.h>

class MatrixSlicingTest : public testing::Test
{
protected:
    //
    // Members
    //virtual void SetUp();
    //virtual void TearDown();
    MatrixSlicer slicer;
};
