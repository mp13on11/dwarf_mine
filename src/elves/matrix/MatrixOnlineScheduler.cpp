#include "Elf.h"
#include "Matrix.h"
#include "MatrixElf.h"
#include "MatrixHelper.h"
#include "MatrixOnlineScheduler.h"
#include "MatrixSlice.h"
#include "MatrixSlicer.h"
#include "MatrixSlicerOnline.h"
#include "common/ProblemStatement.h"

using namespace std;
using MatrixHelper::MatrixPair;

std::vector<MatrixSlice> MatrixOnlineScheduler::sliceDefinitions = std::vector<MatrixSlice>();
std::vector<MatrixSlice>::iterator MatrixOnlineScheduler::currentSliceDefinition = MatrixOnlineScheduler::sliceDefinitions.begin();
map<NodeId, bool> MatrixOnlineScheduler::finishedSlaves = map<NodeId, bool>();

MatrixOnlineScheduler::MatrixOnlineScheduler(const function<ElfPointer()>& factory) :
    MatrixScheduler(factory)
{
    if (!MpiHelper::isMaster()) return;
    for (size_t i = 1; i < MpiHelper::numberOfNodes(); ++i)
        finishedSlaves[NodeId(i)] = false;
}

MatrixOnlineScheduler::~MatrixOnlineScheduler()
{
}

void MatrixOnlineScheduler::doDispatch()
{
    if (MpiHelper::isMaster())
        orchestrateCalculation();
    else
        calculateOnSlave();
}

void MatrixOnlineScheduler::orchestrateCalculation()
{
    sliceInput();
    schedule();
}

void MatrixOnlineScheduler::sliceInput()
{
    MatrixSlicerOnline slicer;
    result = Matrix<float>(left.rows(), right.columns());
    sliceDefinitions = slicer.layout(result.rows(), result.columns(), finishedSlaves.size(), 1);
    currentSliceDefinition = sliceDefinitions.begin();
}

void MatrixOnlineScheduler::schedule()
{
    while (hasSlices() || !haveSlavesFinished())
    {
        NodeId requestingNode;
        requestingNode = MatrixHelper::getNextSliceRequest();
        fetchResultFrom(requestingNode);
        sendNextSlicesTo(requestingNode);
    }
}

void MatrixOnlineScheduler::fetchResultFrom(const NodeId node)
{
    Matrix<float> nodeResult = MatrixHelper::receiveMatrixFrom(node);
    if (nodeResult.empty()) return;
    MatrixSlice& sliceDefinition = getNextSliceDefinitionFor(node);
    sliceDefinition.injectSlice(nodeResult, result);
    sliceDefinition.setNodeId(MpiHelper::MASTER);
}

MatrixSlice& MatrixOnlineScheduler::getNextSliceDefinitionFor(const NodeId node)
{
    for (auto& slice : sliceDefinitions)
        if (slice.getNodeId() == node)
            return slice;
    throw "ERROR: No next slice definition found.";
}

void MatrixOnlineScheduler::sendNextSlicesTo(const NodeId node)
{
    MatrixPair requestedSlices;
    if (hasSlices())
    {
        requestedSlices = sliceMatrices(*currentSliceDefinition);
        (*currentSliceDefinition).setNodeId(node);
        currentSliceDefinition++;
    }
    else
    {
        requestedSlices = MatrixPair(Matrix<float>(0, 0), Matrix<float>(0, 0));
        finishedSlaves[node] = true;
    }
    MatrixHelper::sendMatrixTo(requestedSlices.first, node);
    MatrixHelper::sendMatrixTo(requestedSlices.second, node);
}

bool MatrixOnlineScheduler::hasSlices() const
{
    return currentSliceDefinition != sliceDefinitions.end();
}

bool MatrixOnlineScheduler::haveSlavesFinished() const
{
    for (const auto& slaveState : finishedSlaves)
        if (!slaveState.second)
            return false;
    return true;
}

void MatrixOnlineScheduler::calculateOnSlave()
{
    Matrix<float> left, right, result = Matrix<float>(0, 0);
    calculateNextResult(result, left, right);
    while (!left.empty() && !right.empty())
        calculateNextResult(result, left, right);
}

void MatrixOnlineScheduler::calculateNextResult(
    Matrix<float>& result,
    Matrix<float>& left,
    Matrix<float>& right)
{
    int rank = MpiHelper::rank();
    MatrixHelper::requestNextSlice(rank);
    MatrixHelper::sendMatrixTo(result, MpiHelper::MASTER);
    left = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    right = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    result = elf().multiply(left, right);
}

