#include "MatrixOnlineSchedulingRowwise.h"

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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
int MatrixOnlineSchedulingRowwise::getWorkAmountFor(const NodeId node)
{
    return 1;
}
#pragma GCC diagnostic pop

void MatrixOnlineSchedulingRowwise::reset()
{
}

