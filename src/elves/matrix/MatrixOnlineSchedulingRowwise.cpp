#include "MatrixOnlineSchedulingRowwise.h"
#include "Matrix.h"
#include "MatrixOnlineScheduler.h"

#include <algorithm>

using namespace std;

const int MatrixOnlineSchedulingRowwise::defaultWorkAmount = 1;

vector<MatrixSlice> MatrixOnlineSchedulingRowwise::getSliceDefinitions(
    const Matrix<float>& result,
    const BenchmarkResult& nodeSet)
{
    return slicer.layout(
        result.rows(),
        result.columns(),
        (nodeSet.size() - 1) * defaultWorkAmount,
        1);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
int MatrixOnlineSchedulingRowwise::getLastWorkAmountFor(
    const MatrixOnlineScheduler& scheduler,
    const int node)
{
    return lastWorkAmounts.find(node) != lastWorkAmounts.end() ?
        lastWorkAmounts[node] :
        1;
}
#pragma GCC diagnostic pop

int MatrixOnlineSchedulingRowwise::getNextWorkAmountFor(
    const MatrixOnlineScheduler& scheduler,
    const int node)
{
    const int remainingWorkAmount = scheduler.getRemainingWorkAmount();
    const int nextWorkAmount = max(min(remainingWorkAmount + 1, defaultWorkAmount), 1);
    lastWorkAmounts[node] = nextWorkAmount;
    return nextWorkAmount;
}
   
void MatrixOnlineSchedulingRowwise::reset()
{
}

