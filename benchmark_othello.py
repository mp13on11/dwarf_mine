#!/usr/bin/python

import os
import select
import itertools
import subprocess
import datetime
from math import sqrt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pylab import *
from optparse import OptionParser

import random
import time

DIRECTORY_NAME = "values"
OUTPUT_DIRECTORY_NAME = "diagrams"
ITERATION_STEPS = 3

def collectMeasuresForTrial(trial):
	smp = []
	cuda = []
	
	for fileName in sorted(os.listdir(DIRECTORY_NAME)):
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

def collectMeasuresForRevision(rev):
	smp = {}
	cuda = {}
	
	for fileName in sorted(os.listdir(DIRECTORY_NAME)):
		fileNameParts = fileName.split("_")
		date = fileNameParts[0]
		mode = fileNameParts[1]
		fileTrial = fileNameParts[2]
		fileRev = fileNameParts[3][:-4]
		
		if fileRev == rev:
			if mode == "smp":
				smp[fileTrial] = (date, avgTimeFromFile(fileName))
			if mode == "cuda":
				cuda[fileTrial] = (date, avgTimeFromFile(fileName))
	
	return smp, cuda

def collectData(directoryName, revsToBenchmark):
	timestamps_hash = []
	trials = []

	for fileName in os.listdir(directoryName):
		timestamp, mode, trial, revision = fileName.split("_")
		revision = revision.replace(".txt", "")
		
		if revision in revsToBenchmark:
			if (timestamp, revision) not in timestamps_hash:
				timestamps_hash.append((timestamp, revision))
			if trial not in trials:	
				trials.append(trial)

	return timestamps_hash, trials

def avg(xs):
	return int(sum(xs) / len(xs))

def avgTimeFromFile(fileName):
	return avg([float(line) for line in open(os.path.join(DIRECTORY_NAME, fileName))])

def plotTrialsForRevision(smp, cuda, rev, trials, fileName):
	
	if not os.path.exists(OUTPUT_DIRECTORY_NAME):
		os.makedirs(OUTPUT_DIRECTORY_NAME)

	trials = sorted(trials)

	ind = arange(len(smp))
	
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	ax.plot(ind, [smp[trial][1] for trial in trials], label = "SMP", color = "r")
	ax.plot(ind, [cuda[trial][1] for trial in trials], label = "CUDA", color = "y")
	
	ax.set_ylabel("Runtime")
	ax.set_title("Revision:" + rev)
	ax.set_xticks(ind)
	ax.set_xticklabels(trials)
	
	ax.legend(loc = 2)
	
	pyplot.savefig(os.path.join(OUTPUT_DIRECTORY_NAME,fileName), bbox_inches = "tight")
	
def plotChange(smp, cuda, trial, fileName):
	
	if not os.path.exists(OUTPUT_DIRECTORY_NAME):
		os.makedirs(OUTPUT_DIRECTORY_NAME)

	ind = arange(len(smp))
	width = 0.35
	
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	rects1 = ax.bar(ind, [value for date, value in smp], width, label = "SMP", color = "r")
	rects2 = ax.bar(ind + width, [value for date, value in cuda], width, label = "CUDA", color = "y")
	
	ax.set_ylabel("Runtime")
	ax.set_title("Othello trials:" + trial)
	ax.set_xticks(ind + width)
	ax.set_xticklabels([date for date, value in smp])
	
	ax.legend(loc = 2)
	
	pyplot.savefig(os.path.join(OUTPUT_DIRECTORY_NAME,fileName), bbox_inches = "tight")

def getCurrentRev():
	proc = subprocess.Popen(["git log --pretty=format:'%H_%ct' -n 1"], stdout=subprocess.PIPE, shell=True)
	rev, date = proc.communicate()[0].split("_")

def createFileName(rev, date, mode, numberOfIterations):
	return DIRECTORY_NAME + "/" + str(date) + "_" + mode + "_" + str(numberOfIterations) + "_" + str(rev)+ ".txt"

def getCommandFor(numberOfIterations, mode, rev, date):
	return "./build/src/main/dwarf_mine -m {0} -c montecarlo_tree_search -n 100 -w 10 --montecarlo_trials {1} -q --time_output {2}".format(mode, numberOfIterations, createFileName(rev, date, mode, numberOfIterations))

def changeRevision(revision):
	print "== Checkout revision " + revision + " =="
	proc = subprocess.Popen(["git checkout " + revision], stdout=subprocess.PIPE, shell=True)
	result, err = proc.communicate()
	print "--> " + result

def build():
	print "== Build =="
	proc = subprocess.Popen(["./build.sh"], stdout=subprocess.PIPE, shell=True)
	result, err = proc.communicate()
	print "--> " + result

def setToLatestRevision(branch):
	print "== Return to latest revision =="
	proc = subprocess.Popen(["git checkout " + branch], stdout=subprocess.PIPE, shell=True)
	result, err = proc.communicate()
	print "--> " + result

def getRevisionDate():
	proc = subprocess.Popen(["git log --pretty=format:'%ct' -n 1"], stdout=subprocess.PIPE, shell=True)
	return proc.communicate()[0]

def measure(numberOfIterations, mode, rev):
	
	if not os.path.exists(DIRECTORY_NAME):
		os.makedirs(DIRECTORY_NAME)
	
	changeRevision(rev)
	date = getRevisionDate()
	build()
	command = getCommandFor(numberOfIterations, mode, rev, date)
	print "== EXECUTE COMMAND:", command
	os.system(command)

if __name__ == "__main__":

	parser = OptionParser()
	parser.add_option("-r", dest = "rev_file")
	parser.add_option("-b", dest = "branch")
	(options, args) = parser.parse_args()

	iterations = [10 ** n for n in range(1, ITERATION_STEPS + 1)]
	modes = ["smp", "cuda"]

	# determine the revs to compare
	revsToBenchmark = []	
	if options.rev_file is not None:
			revsToBenchmark = [line.strip() for line in open(options.rev_file)]
	else:
		print "NO INPUT - USE A FILE WITH REVISON HASHES AND PARAMETER '-r'"
		sys.exit(-1)	

	for revision, iteration, mode in itertools.product(revsToBenchmark, iterations, modes):
		print "== Measure revision ", revision, " with ", iteration, " iterations in ", mode, " =="			
		measure(iteration, mode, revision)
		
	timestamp_hashes, trials = collectData(DIRECTORY_NAME, revsToBenchmark)

	for trial in trials:
		smp, cuda = collectMeasuresForTrial(trial)
		plotChange(smp, cuda, trial, "change_" + trial)

	for timestamp_hash in timestamp_hashes:
		smp, cuda = collectMeasuresForRevision(timestamp_hash[1])
		plotTrialsForRevision(smp, cuda, timestamp_hash[0], trials, "rev_" + timestamp_hash[1])

	if options.branch is None:
		print "Use -b to determine the branch!"	
		sys.exit(-1)

	setToLatestRevision(options.branch)
