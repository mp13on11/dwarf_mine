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

#pragma once

#include "Matrix.h"
#include "common/SchedulerTemplate.h"

#include <functional>
#include <vector>

class MatrixElf;
class MatrixSlice;

class MatrixScheduler: public SchedulerTemplate<MatrixElf>
{
public:
    MatrixScheduler(const Communicator& communicator, const std::function<ElfPointer()>& factory);
    virtual ~MatrixScheduler();

    virtual void provideData(std::istream& input);
    virtual void outputData(std::ostream& output);
    virtual void generateData(const DataGenerationParameters& params);

protected:
    Matrix<float> left;
    Matrix<float> right;
    Matrix<float> result;

    virtual bool hasData() const;
    virtual void doDispatch();
    virtual void doSimpleDispatch();

    virtual void orchestrateCalculation();
    virtual void calculateOnSlave();
    virtual void calculateOnMaster(const MatrixSlice& definition, Matrix<float>& result) const;
    virtual Matrix<float> dispatchAndReceive() const;
    virtual void collectResults(const std::vector<MatrixSlice>& slices, Matrix<float>& result) const;
    std::pair<Matrix<float>, Matrix<float>> sliceMatrices(const MatrixSlice& definition) const;

private:
    const MatrixSlice* distributeSlices(const std::vector<MatrixSlice>& slices) const;
