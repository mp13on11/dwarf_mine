/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

#include "MonteCarloFactorizationElf.h"
#include "FactorizationScheduler.h"
#include "common/ProblemStatement.h"

#include <memory>
#include <stdexcept>
#include <vector>

using namespace std;
using namespace std::chrono;

const BigInt DEFAULT_PRODUCT(1649);

FactorizationScheduler::FactorizationScheduler(const Communicator& communicator, const function<ElfPointer()>& factory) :
    SchedulerTemplate(communicator, factory)
{
}

void FactorizationScheduler::doSimpleDispatch()
{
    tie(p, q) = elf().factor(number);
}

void FactorizationScheduler::provideData(istream& input)
{
    input >> number;

    if (input.fail())
        throw runtime_error("Failed to read BigInt from input stream in " __FILE__);
}

void FactorizationScheduler::outputData(ostream& output)
{
    output << p << endl;
    output << q << endl;
}

void FactorizationScheduler::generateData(const DataGenerationParameters&)
{
    number = DEFAULT_PRODUCT;
}

void FactorizationScheduler::doDispatch()
{
    distributeNumber();

    future<BigIntPair> f = async(launch::async,
        &MonteCarloFactorizationElf::factor, &elf(), number
    );

    int rank = distributeFinishedStateRegularly(f);
    elf().stop();
    sendResultToMaster(rank, f);
}

bool FactorizationScheduler::hasData() const
{
    return number != 0;
}

void FactorizationScheduler::distributeNumber()
{
    if (communicator.isMaster())
    {
        string s = number.get_str();
        unsigned long size = s.length();
        communicator->Bcast(&size, 1, MPI::UNSIGNED_LONG, Communicator::MASTER_RANK);
        communicator->Bcast(
                const_cast<char*>(s.c_str()), size,
                MPI::CHAR, Communicator::MASTER_RANK
            );
    }
    else
    {
        unsigned long size = 0;
        communicator->Bcast(&size, 1, MPI::UNSIGNED_LONG, Communicator::MASTER_RANK);
        unique_ptr<char[]> items(new char[size]);
        communicator->Bcast(items.get(), size, MPI::CHAR, Communicator::MASTER_RANK);

        number = BigInt(string(items.get(), size));
    }
}

void FactorizationScheduler::sendResultToMaster(int rank, future<BigIntPair>& f)
{
    if (!communicator.isMaster() && rank != communicator.rank())
        return;

    if (rank == communicator.rank())
    {
        BigIntPair pair = f.get();
        p = pair.first;
        q = pair.second;
    }

    // if the master found the result, nothing more to do
    if(rank == Communicator::MASTER_RANK)
        return;

    // a slave found the result, he has to send it to the master
    if (communicator.isMaster())
    {
        unsigned long sizes[] = { 0, 0 };
        communicator->Recv(sizes, 2, MPI::UNSIGNED_LONG, rank, 1);

        unique_ptr<char[]> first(new char[sizes[0]]);
        unique_ptr<char[]> second(new char[sizes[1]]);

        communicator->Recv(first.get(), sizes[0], MPI::CHAR, rank, 2);
        communicator->Recv(second.get(), sizes[1], MPI::CHAR, rank, 3);

        p = BigInt(string(first.get(), sizes[0]));
        q = BigInt(string(second.get(), sizes[1]));
    }
    else
    {
        string first = p.get_str();
        string second = q.get_str();
        unsigned long sizes[] = { first.length(), second.length() };

        communicator->Send(sizes, 2, MPI::UNSIGNED_LONG, Communicator::MASTER_RANK, 1);
        communicator->Send(first.c_str(), first.size(), MPI::CHAR, Communicator::MASTER_RANK, 2);
        communicator->Send(second.c_str(), second.size(), MPI::CHAR, Communicator::MASTER_RANK, 3);
    }
}

int FactorizationScheduler::distributeFinishedStateRegularly(future<BigIntPair>& f) const
{
    unique_ptr<bool[]> states(new bool[communicator.size()]);

    while (true)
    {
        bool finished = f.wait_for(seconds(5)) == future_status::ready;

        communicator->Allgather(
                &finished, 1, MPI::BOOL,
                states.get(), 1, MPI::BOOL
            );

        for (size_t i=0; i<communicator.size(); i++)
        {
            if (states[i])
                return i;
        }
    }
}
