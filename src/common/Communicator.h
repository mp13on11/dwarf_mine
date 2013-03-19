#pragma once

#include <mpi.h>
#include <stdexcept>
#include <algorithm>

class Communicator
{
public:

    static const int MASTER_RANK = 0;

    Communicator();
    Communicator(const std::vector<double>& weights);

    Communicator createSubCommunicator(std::initializer_list<int> newNodes) const;

    bool isMaster() const;
    int rank() const;
    size_t size() const;
    double weight() const;
    std::vector<double> weights() const;

private:
    MPI::Intracomm _communicator;
    std::vector<double> _weights;

    Communicator(const MPI::Intracomm& communicator, const std::vector<double>& weights);
    void broadcastWeights();
};

Communicator::Communicator() : 
    _communicator(MPI::COMM_WORLD),
    _weights(_communicator.Get_size(), 1.0 / _communicator.Get_size())
{    
}

Communicator::Communicator(const std::vector<double>& weights) : 
    _communicator(MPI::COMM_WORLD),
    _weights(weights)
{    
    if (_weights.size() != _communicator.Get_size())
    {
        throw std::runtime_error("Number of weights differs from number of MPI nodes.");
    }
}

Communicator::Communicator(const MPI::Intracomm& communicator, const std::vector<double>& weights) : 
    _communicator(communicator),
    _weights(weights)
{    
    if (_weights.size() != _communicator.Get_size())
    {
        throw std::runtime_error("Number of weights differs from number of MPI nodes.");
    }
}

void Communicator::broadcastWeights()
{
    _communicator.Bcast(_weights.data(), _weights.size(), MPI::DOUBLE, MASTER_RANK);
}

Communicator Communicator::createSubCommunicator(std::initializer_list<int> newNodes) const
{
    bool isIncluded = std::find(newNodes.begin(), newNodes.end(), rank()) != newNodes.end();
    int color = isIncluded ? 1 : MPI_UNDEFINED;
    auto newMPICommunicator = _communicator.Split(color, 0);
    std::vector<double> newWeights;
    for (int nodeRank : newNodes)
    {
        newWeights.push_back(_weights[nodeRank]);
    }
    return Communicator(newMPICommunicator, newWeights); 
}

bool Communicator::isMaster() const
{
    return rank() == MASTER_RANK;
}

int Communicator::rank() const
{
    return _communicator.Get_rank();
}

size_t Communicator::size() const
{
    return _communicator.Get_size();
}

double Communicator::weight() const
{
    return _weights[rank()];
}

std::vector<double> Communicator::weights() const
{
    return _weights;
}