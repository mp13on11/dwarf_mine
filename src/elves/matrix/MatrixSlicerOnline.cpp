#include "MatrixSlicerOnline.h"
#include "MatrixSlice.h"
#include <cmath>
#include <iostream>

using namespace std;

vector<MatrixSlice> MatrixSlicerOnline::layout(size_t rows, size_t columns, size_t rowparts, size_t colparts)
{
    slices.clear();
    size_t slice_rows = ceil((double)rows / rowparts);
    size_t slice_cols = ((double)columns / colparts);
    for(size_t i=0; i<=columns; i+= slice_cols){
        for(size_t j=0; j<=rows; j+= slice_rows){
            slices.push_back(MatrixSlice{i, j, slice_cols, slice_rows});
        }
    }
    return slices;
}

