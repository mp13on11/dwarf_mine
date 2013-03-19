#pragma once

#include <vector>

template<typename T>
class SparseVector
{
public:
    bool empty() const {return indices.empty();}
    void isSet(T index) const;
    void set(T index);
    void flip(T index);
    std::vector<T> indices;
};
