#include <ext/stdio_filebuf.h>
#include <iostream>
#include <fstream>
#include "gtest/gtest.h"

using namespace std;

double square_root (const double x)
{
    double left = 0;
    double right = x;
    double middle, error;

    do
    {
        middle = (left + right) / 2;
        if(middle*middle > x)
            right = middle;
        else
            left = middle;
        error = (middle*middle - x) * (middle*middle - x);
    }while(error > 1e-20);
    return middle;
}


TEST (SquareRootTest, PositiveNos) { 
    EXPECT_NEAR (182.0, square_root (324.0), 1e-4);
    EXPECT_NEAR (25.4, square_root (645.16), 1e-4);
    EXPECT_NEAR (50.3321, square_root (2533.310224), 1e-4);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}