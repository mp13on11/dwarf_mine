#include "MatrixOnlineSchedulingRowwise.h"
#include "Matrix.h"

using namespace std;

vector<MatrixSlice> MatrixOnlineSchedulingRowwise::getSliceDefinitions(
    const Matrix<float>& result,
    const BenchmarkResult& nodeSet)
{
    return slicer.layout(
        result.rows(),
        result.columns(),
        nodeSet.size(),
        1);
}

int MatrixOnlineSchedulingRowwise::getWorkAmountFor(const int /* node */)
{
    return 1;
}

void MatrixOnlineSchedulingRowwise::reset()
{
}

