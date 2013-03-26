#include "MatrixOnlineSchedulingRowwise.h"
#include "Matrix.h"
#include "MatrixSlicerOnline.h"
#include "common/BenchmarkResults.h"

using namespace std;

size_t MatrixOnlineSchedulingRowwise::getWorkQueueSize()
{
    return 5;
}

vector<MatrixSlice> MatrixOnlineSchedulingRowwise::getSliceDefinitions(
    const Matrix<float>& result,
    const BenchmarkResult& nodeSet)
{
    return slicer.layout(
        result.rows(),
        result.columns(),
        nodeSet.size() * getWorkQueueSize(),
        1);
}
