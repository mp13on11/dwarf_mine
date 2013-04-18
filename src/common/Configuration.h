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

#include <boost/program_options.hpp>
#include <iosfwd>
#include <memory>
#include <string>

class Communicator;
class ProblemStatement;
class Scheduler;
class SchedulerFactory;
struct DataGenerationParameters;

class Configuration
{
public:
    static void printHelp();

    Configuration(int argc, char** argv);

    std::unique_ptr<Scheduler> createScheduler(const Communicator& communicator) const;

    std::unique_ptr<ProblemStatement> createProblemStatement() const;
    std::unique_ptr<ProblemStatement> createGeneratedProblemStatement() const;
    std::unique_ptr<SchedulerFactory> createSchedulerFactory() const;

    size_t warmUps() const;
    size_t iterations() const;
    bool shouldExportConfiguration() const;
    bool shouldImportConfiguration() const;
    bool shouldSkipBenchmark() const;
    std::string importConfigurationFilename() const;
    std::string exportConfigurationFilename() const;
    bool shouldBeQuiet() const;
    bool shouldBeVerbose() const;
    std::string timeOutputFilename() const;
    std::string schedulingStrategy() const;
    bool mpiThreadMultiple() const;

    void validate() const;
    bool shouldPrintHelp() const;

    friend std::ostream& operator<<(std::ostream& s, const Configuration& c);

private:
    std::string mode() const;
    std::string category() const;

    DataGenerationParameters makeDataGenerationParameters() const;

    static boost::program_options::options_description createDescription();

    boost::program_options::options_description description;
    boost::program_options::variables_map variables;

    std::string inputFilename() const;
    std::string outputFilename() const;
    size_t leftMatrixRows() const;
    size_t commonMatrixRowsColumns() const;
    size_t rightMatrixColumns() const;
    size_t leftDigits() const;
    size_t rightDigits() const;
    size_t monteCarloTrials() const;
    bool useFiles() const;
