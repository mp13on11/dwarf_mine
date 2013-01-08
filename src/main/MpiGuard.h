#pragma once

#include <cstddef>

class MpiGuard
{
public:
    static bool isMaster();
    static size_t numberOfNodes();

    MpiGuard(int argc, char** argv);
    MpiGuard(const MpiGuard& copy) = delete;
    MpiGuard(MpiGuard&& move) = delete;
    ~MpiGuard();

    MpiGuard& operator=(const MpiGuard& rhs) = delete;
    MpiGuard& operator=(const MpiGuard&& rhs) = delete;

private:
    static const int MASTER_RANK;
};
