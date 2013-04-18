#pragma once

#include "MatrixSlice.h"
#include "common/BenchmarkResults.h"
#include <list>
#include <vector>

class MatrixSlicerOnline
{
public:
    typedef std::vector<MatrixSlice> SliceList;

    SliceList layout(size_t rows, size_t columns, size_t rowparts, size_t columnparts);

private:
    SliceList slices;
}
;
