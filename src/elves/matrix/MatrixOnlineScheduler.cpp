#include "Elf.h"
#include "Matrix.h"
#include "MatrixElf.h"
#include "MatrixOnlineScheduler.h"
#include "MatrixSlice.h"
#include "MatrixSlicer.h"
#include "MatrixSlicerOnline.h"
#include "common/ProblemStatement.h"

#include <algorithm>
#include <iterator>

using namespace std;
using MatrixHelper::MatrixPair;

std::vector<MatrixSlice> MatrixOnlineScheduler::sliceDefinitions = std::vector<MatrixSlice>();
std::vector<MatrixSlice>::iterator MatrixOnlineScheduler::currentSliceDefinition = MatrixOnlineScheduler::sliceDefinitions.begin();
map<NodeId, bool> MatrixOnlineScheduler::finishedSlaves = map<NodeId, bool>();

MatrixOnlineScheduler::MatrixOnlineScheduler(const function<ElfPointer()>& factory) :
    MatrixScheduler(factory)
{
    if (MpiHelper::isMaster())
        for (size_t i = 1; i < MpiHelper::numberOfNodes(); ++i)
            finishedSlaves[NodeId(i)] = false;
    else
        resultQueue.push_back(Matrix<float>(0, 0));
}

MatrixOnlineScheduler::~MatrixOnlineScheduler()
{
}

void MatrixOnlineScheduler::generateData(const DataGenerationParameters& params)
{
    MatrixScheduler::generateData(params);
    // TODO: Adopt params.schedulingStrategy;
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
    sliceDefinitions = slicer.layout(
        result.rows(),
        result.columns(),
        workAmount * finishedSlaves.size(),
        1);
    currentSliceDefinition = sliceDefinitions.begin();
}

void MatrixOnlineScheduler::schedule()
{
    while (hasSlices() || !haveSlavesFinished())
    {
        cout << "Awaiting request" << endl;
        const NodeId requestingNode = MatrixHelper::waitForSlicesRequest();
        cout << "Node " << requestingNode << " requested" << endl;
        const int workAmount = getWorkAmountFor(requestingNode);
        fetchResultsFrom(requestingNode, workAmount);
        sendNextSlicesTo(requestingNode, workAmount);
    }
}

void MatrixOnlineScheduler::fetchResultsFrom(const NodeId node, const int workAmount)
{
    cout << "\tFetching results from node " << node << endl;
    for (int i = 0; i < workAmount; ++i)
    {
        Matrix<float> nodeResult = MatrixHelper::receiveMatrixFrom(node);
        cout << "\t\tReceived result " << i << " from node " << node << endl;
        if (nodeResult.empty()) return;
        MatrixSlice& sliceDefinition = getNextSliceDefinitionFor(node);
        sliceDefinition.injectSlice(nodeResult, result);
        sliceDefinition.setNodeId(MpiHelper::MASTER);
        cout << "\t\tMerged result from node " << node << endl;
    }
}

int MatrixOnlineScheduler::getWorkAmountFor(const NodeId node) const
{
    return workAmount * (node/node); // Temporary b/c warnings = errors :)
}

MatrixSlice& MatrixOnlineScheduler::getNextSliceDefinitionFor(const NodeId node)
{
    for (auto& slice : sliceDefinitions)
        if (slice.getNodeId() == node)
            return slice;
    throw "ERROR: No next slice definition found.";
}

void MatrixOnlineScheduler::sendNextSlicesTo(const NodeId node, const int workAmount)
{
    cout << "\tSending work to node " << node << endl;
    sendWorkAmountTo(node, workAmount);
    for (int i = 0; i < workAmount; ++i)
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
        cout << "\t\tSending work " << i << (finishedSlaves[node] ? " (empty)" : "") << endl;
        MatrixHelper::sendMatrixTo(requestedSlices.first, node);
        MatrixHelper::sendMatrixTo(requestedSlices.second, node);
        if (finishedSlaves[node]) { cout << "\t\tNode " << node << " finished." << endl; return;}
    }
}

void MatrixOnlineScheduler::sendWorkAmountTo(const NodeId node, const int workAmount)
{
    const int actualWorkAmount = min(getRemainingWorkAmount() + 1, workAmount);
    cout << "\tWork amount is " << actualWorkAmount << endl;
    MatrixHelper::sendWorkAmountTo(node, actualWorkAmount);
}

int MatrixOnlineScheduler::getRemainingWorkAmount()
{
    return distance(currentSliceDefinition, sliceDefinitions.end());
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
    file.open("/tmp/log");
    while (hasToWork())
    {
        file << "Has to work" << endl;
        workQueue.clear();
        file << "\tInitiating communication" << endl;
        initiateCommunication();
        file << "\tSending results" << endl;
        sendResults();
        resultQueue.clear();
        file << "\tReceiving work" << endl;
        receiveWork();
        file << "\tCalculating" << endl;
        doWork();
    }
    file << "Finished work" << endl;
    workQueue.clear();
    file << "\tInitiating communication" << endl;
    initiateCommunication();
    file << "\tSending results" << endl;
    sendResults();
    resultQueue.clear();
    file.close();
}

bool MatrixOnlineScheduler::hasToWork()
{
    file << "Work queue empty? " << (workQueue.empty() ? "Yes" : "No") << endl;
    if (!workQueue.empty())
        file << "Received empty input? " << ((workQueue.back().first.empty() || workQueue.back().second.empty()) ? "Yes" : "No") << endl;
    return workQueue.empty()
        || (!workQueue.back().first.empty() && !workQueue.back().second.empty());
}

void MatrixOnlineScheduler::initiateCommunication() const
{
    MatrixHelper::requestNextSlices(MpiHelper::rank());
}

void MatrixOnlineScheduler::sendResults()
{
    for (const auto& result : resultQueue)
    {
        file << "\t\tSending result " << &result << endl;
        MatrixHelper::sendMatrixTo(result, MpiHelper::MASTER);
    }
}

void MatrixOnlineScheduler::receiveWork()
{
    const int workAmount = MatrixHelper::receiveWorkAmountFrom(MpiHelper::MASTER);
    for (int i = 0; i < workAmount; ++i)
    {
        file << "\t\tReceiving work " << i+1 << "/" << workAmount << endl;
        Matrix<float> left = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
        Matrix<float> right = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
        workQueue.push_back({left, right});
        file << "\t\tWork received" << endl;
    }
}

void MatrixOnlineScheduler::doWork()
{
    for (const auto& work : workQueue)
        resultQueue.push_back(elf().multiply(work.first, work.second));
}

