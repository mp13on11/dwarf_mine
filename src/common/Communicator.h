#pragma once

#include "BenchmarkResults.h"

#include <mpi.h>
#include <string>
#include <vector>

/**
 * Wraps an MPI communicator (like MPI::COMM_WORLD).
 *
 * The public constructors will block until all MPI processes have called
 * one of them. The default constructors assigns all nodes an equal weight
 * (i.e. 1 / group size). The other public constructor allows you to specify
 * each node's weight.
 *
 * NOTE:
 *
 * The constructor called by the master process broadcasts its weights to all
 * other MPI processes and overwrites their weights.
 * Thus, you normally want to call
 *
 *      std::vector<double> weights = ... // calculate weights somehow
 *      Communicator(weights)
 *
 * in the master process and
 *
 *      Communicator()
 *
 * in the slave processes.
 */
class Communicator
{
public:
    struct Node;

    static const int MASTER_RANK;

    /**
     * Creates a communicator which wraps the MPI::COMM_WORLD.
     * If this is called by the master, all nodes will have the same weight
     * (1 / group size).
     */
    Communicator();

    /**
     * Creates a communicator which wraps the MPI::COMM_WORLD.
     * If this is called by the master, the specified weights will be broadcasted
     * to the other MPI processes (overriding their weights).
     * The weights will be normalized (sum of weights == 1) before broadcasting.
     *
     * Raises an exception if the number of weights does not match the number
     * of MPI processes.
     * Raises an exception if a negative weight is encountered.
     */
    Communicator(const std::vector<double>& weights);

    /**
     * Creates a new communicator that only contains the specified nodes.
     * This call blocks until all MPI processes that are a member of this
     * communicator have called it.
     *
     * All MPI processes should pass the same ranks for the new communicator
     * (otherwise: undefined behavior).
     *
     * This method returns an invalid communicator (i.e. a communicator that
     * wraps MPI::COMM_NULL) for all MPI processes that are not included in the
     * newNodes vector.
     */
    Communicator createSubCommunicator(std::initializer_list<Node> newNodes) const;

    /**
     * Returns false if this communicator wraps MPI::COMM_NULL and thus cannot
     * be used for sending, receiving, broadcasting and other stuff.
     *
     * NOTE:
     * 
     * Calling _ANY_ method except isValid() on an invalid communicator
     * will cause the corresponding MPI process to terminate.
     */
    bool isValid() const;

    /**
     * Returns true if this communicator wraps MPI::COMM_WORLD, i.e. all MPI
     * processes can be reached with this communicator.
     */
    bool isWorld() const;

    /**
     * Returns whether the calling MPI process is the master in this communicator.
     */
    bool isMaster() const;

    /**
     * Returns the rank of the calling MPI process in this communicator.
     */
    int rank() const;

    /**
     * Returns the number of MPI processes in this communicator.
     */
    size_t size() const;

    /**
     * Returns the weight of the calling MPI process in this communicator.
     *
     * The weights of all processes in every communicator sum up to 1.
     */
    double weight() const;

    /**
     * Returns the weights of all MPI processes in this communicator.
     * This returns valid weights in all MPI processes.
     */
    std::vector<double> weights() const;

    /**
     * Returns a map from node rank to weight of all MPI processes in this
     * communicator.
     * This returns valid weights in all MPI processes.
     */
    BenchmarkResult nodeSet() const;

    /**
     * Allows the use of the MPI::Intracomm methods.
     */
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

inline BenchmarkResult Communicator::nodeSet() const
{
    BenchmarkResult nodeSet;

    for (size_t i=0; i<_weights.size(); ++i)
        nodeSet[i] = _weights[i];

    return nodeSet;
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