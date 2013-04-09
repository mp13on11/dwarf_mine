#!/usr/bin/python
import os
import sys
from datetime import datetime
from optparse import (OptionParser,BadOptionError,AmbiguousOptionError)


base_path = os.popen("pwd").read()
# stolen from http://stackoverflow.com/questions/1885161/how-can-i-get-optparses-optionparser-to-ignore-invalid-options
class PassThroughOptionParser(OptionParser):

    """
    An unknown option pass-through implementation of OptionParser.

    When unknown arguments are encountered, bundle with largs and try again,
    until rargs is depleted.  

    sys.exit(status) will still be called if a known argument is passed
    incorrectly (e.g. missing arguments or bad argument types, etc.)        
    """
    def _process_args(self, largs, rargs, values):
        while rargs:
            try:
                OptionParser._process_args(self,largs,rargs,values)
            except (AttributeError, BadOptionError), e:
		pass

parser = PassThroughOptionParser()
parser.add_option("--base_path", help="Path to the build directory", dest="base_path", default=base_path)
parser.add_option("--build", help="Build directory", dest="build", default="build")
parser.add_option("--app_file", help="Name of the appfile", dest="app_file", default="benchmark_file")
(options, args) = parser.parse_args()


appfile = open(options.app_file, "w+")

passed_arguments = []
skip = False;
for i in range(len(sys.argv) -1):
	if sys.argv[i + 1] == "--base_path" \
		or sys.argv[i + 1] == "--build" \
		or sys.argv[i + 1] == "--app_file":
		skip = True
		continue
	if not skip:
		passed_arguments.append(sys.argv[i + 1])
	skip = False

executable = options.base_path.rstrip("\n")+"/"+options.build.rstrip("\n")+"/src/main/dwarf_mine"
nodes = [
	("bigdwarf", "smp"),
	("bigdwarf", "cuda"),
	("quadcore1", "smp"),
	("quadcore1", "cuda"),
	("quadcore2", "smp"),
#	("quadcore2", "cuda"),
#	("quadcore3", "smp"),
#	("quadcore3", "cuda")
]

for configuration in nodes:
	appfile.write("-host "+configuration[0]+" -np 1 "+executable+" -m "+configuration[1]+" "+" ".join(passed_arguments)+" \n")

appfile.close();


