#pragma once

#include <mpi.h>
#include <string>
#include <vector>

class Communicator
{
public:
    struct Node;

    static const int MASTER_RANK;

    Communicator();
    Communicator(const std::vector<double>& weights);

    Communicator createSubCommunicator(std::initializer_list<Node> newNodes) const;

    bool isValid() const;
    bool isWorld() const;
    bool isMaster() const;
    int rank() const;
    size_t size() const;
    double weight() const;
    std::vector<double> weights() const;

    MPI::Intracomm* operator->() const;

private:
    static std::vector<double> normalize(const std::vector<double>& weights);
    static std::string negativeWeightMessage(double weight, int rank);

    mutable MPI::Intracomm _communicator;
    std::vector<double> _weights;

    Communicator(const MPI::Intracomm& communicator, const std::vector<double>& weights);

    void validateWeights() const;
    void broadcastWeights();
    std::vector<double> calculateNewWeightsFor(const std::vector<Node>& newNodes) const;
};

struct Communicator::Node
{
    int rank;
    double weight;

    Node(int rank);
    Node(int rank, double weight);

    bool operator!=(const Communicator::Node &other) const;
    bool operator==(const Communicator::Node &other) const;
    bool operator<(const Communicator::Node &other) const;
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

inline Communicator::Node::Node(int rank)
    : rank(rank), weight(1.0)
{
}

inline Communicator::Node::Node(int rank, double weight)
    : rank(rank), weight(weight)
{
}

inline bool Communicator::Node::operator!=(const Communicator::Node &other) const
{
    return !(*this == other);
}

inline bool Communicator::Node::operator==(const Communicator::Node &other) const
{
    return rank == other.rank;
}

inline bool Communicator::Node::operator<(const Communicator::Node &other) const
{
    return rank < other.rank;
}