#include "Communicator.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

using namespace std;

template<typename T, typename S>
vector<T> distinctValues(const S&);

const int Communicator::MASTER_RANK = 0;

Communicator::Communicator() : 
    _communicator(MPI::COMM_WORLD),
    _weights(_communicator.Get_size(), 1.0 / _communicator.Get_size())
{    
    broadcastWeights();
}

Communicator::Communicator(const vector<double>& weights) : 
    _communicator(MPI::COMM_WORLD),
    _weights(weights)
{
    validateWeightCount();
    broadcastWeights();
}

Communicator::Communicator(const MPI::Intracomm& communicator, const vector<double>& weights) : 
    _communicator(communicator),
    _weights(weights)
{
    validateWeightCount();
}

void Communicator::broadcastWeights()
{
    _communicator.Bcast(_weights.data(), _weights.size(), MPI::DOUBLE, MASTER_RANK);
}

Communicator Communicator::createSubCommunicator(initializer_list<int> notNecessarilyDistinctNewNodes) const
{
    auto newNodes = distinctValues<int>(notNecessarilyDistinctNewNodes);
    stringstream nodes;
    for (auto n : newNodes)
    {
        nodes << " " << n;
    }
    cout << nodes.str() << endl;

    bool isIncluded = find(newNodes.begin(), newNodes.end(), rank()) != newNodes.end();
    int color = isIncluded ? 1 : MPI_UNDEFINED;
    auto newMPICommunicator = _communicator.Split(color, 0);

    if (newMPICommunicator == MPI::COMM_NULL)
    {
        return Communicator(newMPICommunicator, {}); 
    }

    vector<double> newWeights;
    double newWeightsSum = 0;
    for (int nodeRank : newNodes)
    {
        newWeights.push_back(_weights[nodeRank]);
        newWeightsSum += nodeRank;
    }
    for(size_t i=0; i<newWeights.size(); i++)
    {
        newWeights[i] /= newWeightsSum;
    }
    return Communicator(newMPICommunicator, newWeights); 
}

void Communicator::validateWeightCount() const
{
    if (isValid())
    {
        if (_weights.size() != static_cast<size_t>(_communicator.Get_size()))
        {
            stringstream stream;
            stream << "Number of weights (" << _weights.size()
                << ") differs from number of MPI nodes ("
                << _communicator.Get_size() << ").";

            throw runtime_error(stream.str());
        }
    }
    else if (!_weights.empty())
    {
        throw runtime_error("Received weights for a null communicator.");
    }
}

template<typename T, typename S>
vector<T> distinctValues(const S& init)
{
    vector<T> values(init);
    sort(values.begin(), values.end());
    auto newEnd = unique(values.begin(), values.end());
    return {values.begin(), newEnd};
}