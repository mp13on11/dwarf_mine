#include <gtest/gtest.h>

template<typename t>
class Matrix;

testing::AssertionResult AreMatricesEquals(Matrix<float> expected, Matrix<float>actual, float delta);
testing::AssertionResult AreMatricesEquals(Matrix<float> a, Matrix<float>b);