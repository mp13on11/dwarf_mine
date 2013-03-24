#include "MatrixOnlineSchedulingRowwise.h"
#include "Matrix.h"
#include "MatrixOnlineScheduler.h"

#include <algorithm>

using namespace std;

const int MatrixOnlineSchedulingRowwise::defaultWorkAmount = 3;

vector<MatrixSlice> MatrixOnlineSchedulingRowwise::getSliceDefinitions(
    const Matrix<float>& result,
    const BenchmarkResult& nodeSet)
{
    return slicer.layout(
        result.rows(),
        result.columns(),
        nodeSet.size() * defaultWorkAmount,
        1);
}
