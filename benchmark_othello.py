import os
import subprocess
import datetime
from math import sqrt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pylab import *

import calendar

DIRECTORY_NAME = "values2"
ITERATIONS = 5

def collectMeasuresForTrial(trial):
	smp = []
	cuda = []
	
	for fileName in sorted(os.listdir(DIRECTORY_NAME)):
		print fileName
		fileNameParts = fileName.split("_")
		date = fileNameParts[0]
		mode = fileNameParts[1]
		fileTrial = fileNameParts[2]
		
		if fileTrial == trial:
			if mode == "smp":
				smp.append((date, avgTimeFromFile(fileName)))
			if mode == "cuda":
				cuda.append((date, avgTimeFromFile(fileName)))
			
	return smp, cuda

def collectMeasuresForRevision(directoryName):
	smp = []
	cuda = []
	return smp, cuda

def collectData(directoryName):
	timestamps_hash = []
	trials = []

	for fileName in os.listdir(directoryName):
		timestamp, mode, trial, revision = fileName.split("_")
		revision = revision.replace(".txt", "")
		
		timestamps_hash.append((timestamp, revision))
		trials.append(trial)

	return set(timestamps_hash), set(trials)

def avg(xs):
	#print "===", xs
	return int(sum(xs) / len(xs))

def avgTimeFromFile(fileName):
	#print "==", fileName
	return avg([float(line) for line in open(os.path.join(DIRECTORY_NAME, fileName))])

def plotChange(smp, cuda, trial, fileName):
	
	pyplot.title("Othello Trials: " + trial)
	pyplot.xlabel('Revision')
	pyplot.ylabel('Runtime')
	
	pyplot.xticks(arange(len(smp)), [date for date, value in smp])
	pyplot.plot([value for date, value in smp])
	pyplot.plot([value for date, value in cuda])
	pyplot.legend(["SMP", "CUDA"])

	pyplot.savefig(fileName)

def getCurrentRev():
	proc = subprocess.Popen(["git log --pretty=format:'%H_%ct' -n 1"], stdout=subprocess.PIPE, shell=True)
	return proc.communicate()[0]

def createFileName(rev, mode, numberOfIterations, directory):
	return directory + "/" + mode + "_" + str(numberOfIterations) + "_" + str(rev)+ ".txt"

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
	modes = ["smp", "cuda"]

	for i in range(1, ITERATIONS):
		for mode in modes:
			pass

	timestamp_hashes, trials = collectData(DIRECTORY_NAME)

	for trial in trials:
		smp, cuda = collectMeasuresForTrial(trial)
		plotChange(smp, cuda, trial, "change_" + trial)

	for timestamp_hash in timestamp_hashes:
		smp, cuda = collectMeasuresForRevision(timestamp_hash)

	#for r,d,files in os.walk(DIRECTORY_NAME):
	#	revision_dates = []
#
#		for f in files:
#			date = f.split("_")[1]
#			if date not in revision_dates:
#				revision_dates.append(date)
#			threadTimes = timesFromFile(os.path.join(DIRECTORY_NAME, f))
#

#	print revision_dates