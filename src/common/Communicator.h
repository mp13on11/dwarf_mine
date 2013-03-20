#pragma once

#include <mpi.h>
#include <vector>

class Communicator
{
public:
    static const int MASTER_RANK;

    Communicator();
    Communicator(const std::vector<double>& weights);

    Communicator createSubCommunicator(std::initializer_list<int> newNodes) const;

    bool isValid() const;
    bool isWorld() const;
    bool isMaster() const;
    int rank() const;
    size_t size() const;
    double weight() const;
    std::vector<double> weights() const;

    MPI::Intracomm* operator->() const;

private:
    mutable MPI::Intracomm _communicator;
    std::vector<double> _weights;

    Communicator(const MPI::Intracomm& communicator, const std::vector<double>& weights);

    void validateWeightCount() const;
    void broadcastWeights();
};

inline MPI::Intracomm* Communicator::operator->() const
{
    return & _communicator;
}

inline bool Communicator::isValid() const
{
    return _communicator != MPI::COMM_NULL;
}

inline bool Communicator::isWorld() const
{
    return _communicator == MPI::COMM_WORLD;
}

inline bool Communicator::isMaster() const
{
    return rank() == MASTER_RANK;
}

inline int Communicator::rank() const
{
    return _communicator.Get_rank();
}

inline size_t Communicator::size() const
{
    return _communicator.Get_size();
}

inline double Communicator::weight() const
{
    return _weights[rank()];
}

inline std::vector<double> Communicator::weights() const
{
    return _weights;
}