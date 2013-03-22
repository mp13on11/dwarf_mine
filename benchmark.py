import os
import subprocess
import datetime
from math import sqrt
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pylab import *

DIRECTORY_NAME = "values"

def avg(xs):
    return sum(xs) / len(xs)

def timesFromFile(fileName):
    return [float(line) for line in open(fileName)]

def plotChange(threadTimes, fileName):
	chunksPerThread = dict({1:threadTimes})
	fig = pyplot.figure(figsize=(20, 10))

	pyplot.title('Othello')
	pyplot.xlabel('Run')
	pyplot.ylabel('Runtime')

#	threadChunk = chunksPerThread[key]

#	avgSpeedUps = [avg(threadChunk)]
#	xticks = threadChunk
#	pyplot.xticks(xticks)

#	print xticks
#	print avgSpeedUps
#	pyplot.plot(xticks, avgSpeedUps, '-')


#	pyplot.legend(["#Thread {0}".format(key) for key in chunksPerThread.keys()])
#	ylim(0)
#	xlim(1)
	# pyplot.show()

	pyplot.savefig(fileName, dpi=80)

def getCurrentRev():
	proc = subprocess.Popen(["git log --pretty=format:'%H_%ct' -n 1"], stdout=subprocess.PIPE, shell=True)
	(out, err) = proc.communicate()
	return out

def createFileName(rev, mode, numberOfIterations, directory):
	return directory+"/"+str(rev)+"_"+mode+"_"+str(numberOfIterations)+".txt"

def getCommandFor(directory, numberOfIterations, mode):
	rev = getCurrentRev()
	return "./build/src/main/dwarf_mine -m {0} -c montecarlo_tree_search -n 100 -w 10 --montecarlo_trials {1} -q --time_output {2}".format(mode, numberOfIterations, createFileName(rev, mode, numberOfIterations, directory))

def meassure(numberOfIterations, mode):
	
	if not os.path.exists(DIRECTORY_NAME):
		os.makedirs(DIRECTORY_NAME)
	command = getCommandFor(DIRECTORY_NAME, numberOfIterations, mode,)
	print command
	value = os.system(command)

if __name__ == "__main__":
	start = 10
	iterations = 5
	modes = ["smp", "cuda"]

	for i in range(1, iterations):
		for mode in modes:
			pass
			#values = meassure(start ** i, mode)

	for r,d,files in os.walk(DIRECTORY_NAME):
		for f in files:
			threadTimes = timesFromFile(os.path.join(DIRECTORY_NAME, f))
			plotChange(threadTimes, f.replace(".txt", ".png"))