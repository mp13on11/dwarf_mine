#pragma once

#include <cstddef>

typedef int NodeId;

class MpiHelper
{
public:
    static const NodeId MASTER;

    static bool isMaster();
    static bool isMaster(NodeId id);
    static NodeId rank();
    static size_t numberOfNodes();

    MpiHelper() = delete;
    MpiHelper(const MpiHelper& copy) = delete;
    MpiHelper(MpiHelper&& move) = delete;
    ~MpiHelper() = delete;

    MpiHelper& operator=(const MpiHelper &rhs) = delete;
    MpiHelper& operator=(MpiHelper&& rhs) = delete;
};
