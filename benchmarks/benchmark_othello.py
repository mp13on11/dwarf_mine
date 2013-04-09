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
	return rev, date

def createFileName(rev, date, mode, numberOfIterations):
	return DIRECTORY_NAME + "/" + str(date) + "_" + mode + "_" + str(numberOfIterations) + "_" + str(rev)+ ".txt"

def getCommandFor(numberOfIterations, mode, rev, date):
	return "./build_"+rev+"/src/main/dwarf_mine -m {0} -c montecarlo_tree_search -n 100 -w 10 --montecarlo_trials {1} -q --time_output {2}".format(mode, numberOfIterations, createFileName(rev, date, mode, numberOfIterations))

def changeRevision(revision):
	print "== Checkout revision " + revision + " =="
	proc = subprocess.Popen(["git checkout " + revision], stdout=subprocess.PIPE, shell=True)
	result, err = proc.communicate()
	print "--> " + result

def build(build_dir):
	
	print "== Prepare Build Directory " + build_dir + " =="
	proc = subprocess.Popen(["rm -rf " + build_dir], stdout=subprocess.PIPE, shell=True)
	proc.communicate()

	proc = subprocess.Popen(["mkdir " + build_dir], stdout=subprocess.PIPE, shell=True)
	proc.communicate()

	print "== Change Build Type to Release =="
	proc = subprocess.Popen(['cd ' + build_dir + ' && cmake -G "Unix Makefiles" ..'], stdout=subprocess.PIPE, shell=True)
	result, err = proc.communicate()
	print "--> " + result

	proc = subprocess.Popen(["cd .."], stdout=subprocess.PIPE, shell=True)
	proc.communicate()

	proc = subprocess.Popen(['cmake -D CMAKE_BUILD_TYPE=Release ' + build_dir], stdout=subprocess.PIPE, shell=True)
	result, err = proc.communicate()
	print "--> " + result

	print "== Build =="	
	proc = subprocess.Popen(["make -j -C " + build_dir], stdout=subprocess.PIPE, shell=True)
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
	
	date = getRevisionDate()
	
	command = getCommandFor(numberOfIterations, mode, rev, date)
	print "== EXECUTE COMMAND:", command
	os.system(command)

def plotGraphForMode(mode, trials, translation_file):
	
	trials = sorted(trials)
	revisions = {}
	f = open(translation_file, "r")
	for line in f:
		print line
		hash, translation = line.split(" ")
		revisions[hash] = translation
	
	data = {}
	for revision in revisions.keys():
		values = []
		for t in trials:
			for fileName in os.listdir(DIRECTORY_NAME):
				print fileName
				timestamp, m, trial, rev = fileName.split("_")
				rev = rev.replace(".txt", "")
				
				print "Mode:", mode , " m:", m
				if (rev == revision) and (t == trial) and (mode == m):
					values. append(avgTimeFromFile(fileName))
		data[revision] = values
	
	if not os.path.exists(OUTPUT_DIRECTORY_NAME):
		os.makedirs(OUTPUT_DIRECTORY_NAME)

	ind = arange(len(trials))
	
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	print data

	for key in data.keys():
		values = data[key]
		print values
		ax.plot(ind, values, label = revisions[key])
	
	ax.set_ylabel("Runtime")
	ax.set_xlabel("Trial")
	ax.set_xticks(ind)
	ax.set_xticklabels(trials)
	
	ax.legend(loc = 2)
	
	pyplot.savefig(os.path.join(OUTPUT_DIRECTORY_NAME,mode), bbox_inches = "tight")

if __name__ == "__main__":

	parser = OptionParser()
	parser.add_option("--with_measurements", action="store_true", dest = "with_measurements")
	parser.add_option("-e", type = "int", dest = "exponents_10", default = 3)
	parser.add_option("-r", dest = "rev_file")
	parser.add_option("-b", dest = "branch")
	(options, args) = parser.parse_args()

	iterations = [10 ** n for n in range(1, options.exponents_10 + 1)]
#	modes = ["smp", "cuda"]
	modes = ["cuda"]

	# determine the revs to compare
	revsToBenchmark = []	
	translations = []
	if options.rev_file is not None:
			for line in open(options.rev_file):
				r, t = line.split(" ")
				revsToBenchmark.append(r)
				translations.append(t.strip())
	else:
		print "NO INPUT - USE A FILE WITH REVISON HASHES AND PARAMETER '-r'"
		sys.exit(-1)	

	currentRev = getCurrentRev()[0]

	if options.with_measurements:
		for revision in revsToBenchmark:
			changeRevision(revision)
			build("build_"+revision)	
			for iteration, mode in itertools.product(iterations, modes):
				print "== Measure revision ", revision, " with ", iteration, " iterations in ", mode, "==="			
				measure(iteration, mode, revision)
		
	timestamp_hashes, trials = collectData(DIRECTORY_NAME, revsToBenchmark)

	#for trial in trials:
	#	smp, cuda = collectMeasuresForTrial(trial)
	#	plotChange(smp, cuda, trial, "change_" + trial)

	#for timestamp_hash in timestamp_hashes:
	#	smp, cuda = collectMeasuresForRevision(timestamp_hash[1])
	#	plotTrialsForRevision(smp, cuda, timestamp_hash[0], trials, "rev_" + timestamp_hash[1])

	#for mode in modes:
	#	plotGraphForMode(trials, mode, mode)
	
	plotGraphForMode("cuda", trials, options.rev_file)
	
	#if options.branch is None:
	#	print "Use -b to determine the branch!"	
	#	sys.exit(-1)

	#setToLatestRevision(options.branch)
	changeRevision(currentRev)

