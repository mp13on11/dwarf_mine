#pragma once

#include <map>
#include <iosfwd>

typedef double Rating; // TODO
typedef std::pair<int, Rating> NodeRating;

class BenchmarkResult
{
public:
    typedef std::map<int, Rating>::const_iterator const_iterator;
    typedef std::map<int, Rating>::iterator iterator;
    typedef std::map<int, Rating>::value_type value_type;

    BenchmarkResult();
    BenchmarkResult(const std::map<int,Rating>& ratings);
    BenchmarkResult(std::initializer_list<value_type> init);

    Rating& operator[](const int& nodeId);
    const_iterator begin() const;
    const_iterator end() const;
    iterator begin();
    iterator end();

    size_t size() const;
    bool empty() const;

private:
    std::map<int, Rating> _ratings;
};

std::ostream& operator<<(std::ostream& o, const BenchmarkResult& results);


inline BenchmarkResult::BenchmarkResult()
{
}

inline BenchmarkResult::BenchmarkResult(const std::map<int,Rating>& ratings) : 
    _ratings(ratings) 
{
}

inline BenchmarkResult::BenchmarkResult(std::initializer_list<value_type> init) :
    _ratings(init) 
{
}

inline BenchmarkResult::const_iterator BenchmarkResult::begin() const
{
    return _ratings.begin();
}

inline BenchmarkResult::const_iterator BenchmarkResult::end() const
{
    return _ratings.end();
}

inline BenchmarkResult::iterator BenchmarkResult::begin()
{
    return _ratings.begin();
}

inline BenchmarkResult::iterator BenchmarkResult::end()
{
    return _ratings.end();
}

inline size_t BenchmarkResult::size() const 
{
    return _ratings.size();
}

inline Rating& BenchmarkResult::operator[](const int& nodeId)
{
    return _ratings[nodeId];
}

inline bool BenchmarkResult::empty() const
{
    return _ratings.empty();
}