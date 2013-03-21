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
    _weights(normalize(weights))
{
    validateWeights();
    broadcastWeights();
}

Communicator::Communicator(const MPI::Intracomm& communicator, const vector<double>& weights) : 
    _communicator(communicator),
    _weights(normalize(weights))
{
    validateWeights();
    broadcastWeights();
}

void Communicator::broadcastWeights()
{
    if (!isValid())
        return;
    
    _communicator.Bcast(_weights.data(), _weights.size(), MPI::DOUBLE, MASTER_RANK);
}

Communicator Communicator::createSubCommunicator(initializer_list<Node> notNecessarilyDistinctNewNodes) const
{
    auto newNodes = distinctValues<Node>(notNecessarilyDistinctNewNodes);

    bool isIncluded = find(newNodes.begin(), newNodes.end(), Node(rank())) != newNodes.end();
    int color = isIncluded ? 1 : MPI_UNDEFINED;
    auto newMPICommunicator = _communicator.Split(color, 0);

    if (newMPICommunicator == MPI::COMM_NULL)
    {
        return Communicator(newMPICommunicator, {}); 
    }

    return Communicator(newMPICommunicator, calculateNewWeightsFor(newNodes)); 
}

vector<double> Communicator::calculateNewWeightsFor(const vector<Node>& newNodes) const
{
    vector<double> newWeights;

    for (Node node : newNodes)
    {
        if (node.weight < 0)
            throw runtime_error(negativeWeightMessage(node.weight, node.rank));

        newWeights.push_back(_weights[node.rank] * node.weight);
    }

    cout << endl;

    return newWeights;
}

void Communicator::validateWeights() const
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

        for (size_t i=0; i<_weights.size(); ++i)
        {
            if (_weights[i] < 0)
                throw runtime_error(negativeWeightMessage(_weights[i], i));
        }
    }
    else if (!_weights.empty())
    {
        throw runtime_error("Received weights for a null communicator.");
    }
}

string Communicator::negativeWeightMessage(double weight, int rank)
{
    stringstream stream;
    stream << "Detected negative weight (" << weight
        << ") for node " << rank << ".";

    return stream.str();
}

vector<double> Communicator::normalize(const vector<double>& weights)
{
    double sum = accumulate(weights.begin(), weights.end(), 0.0);

    if (sum == 0)
        return weights;

    vector<double> normalized(weights);

    for (size_t i=0; i<weights.size(); ++i)
    {
        normalized[i] /= sum;
    }

    return normalized;
}

template<typename T, typename S>
vector<T> distinctValues(const S& init)
{
    vector<T> values(init);
    sort(values.begin(), values.end());
    auto newEnd = unique(values.begin(), values.end());
    return {values.begin(), newEnd};
}