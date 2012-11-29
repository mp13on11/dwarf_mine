#include "Arguments.h"
#include "InvalidCommandLineException.h"

#include <boost/lexical_cast.hpp>
#include <sstream>

using namespace std;

Arguments::Arguments() :
        _iterations(1)
{
}

Arguments::Arguments(int argc, const char* argv[]) :
        _program(argv[0]), _iterations(1)
{
    vector<string> args(argv + 1, argv + argc);

    if (args.size() == 0)
        throw InvalidCommandLineException();
    if (args[0] == "--iterations" && args.size() == 1)
        throw InvalidCommandLineException("missing argument to --iterations");

    if (args[0] == "--iterations")
    {
        _iterations = boost::lexical_cast<size_t>(args[1]);
        args.erase(args.begin());
        args.erase(args.begin());
    }

    _inputs = vector<string>(args.begin(), args.end() - 1);
    _output = args.back();
}

void Arguments::printUsage(const string& program, ostream& out)
{
    out << "Usage: " << program << "[<options>] <input file>... <output file>" << endl;
    out << "\toptions: --iterations <n>\tset the number of iterations (default: 1)" << endl;
}

string Arguments::toString() const
{
    stringstream stream;
    stream << _program;

    if (_iterations != 1)
        stream << " --iterations " << _iterations;

    for (const string &input : _inputs)
        stream << " " << input;

    stream << " " << _output;

    return stream.str();
}
