#!/usr/bin/python
import os
import sys
from math import sqrt
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from datetime import datetime


def times_from_file(file_name):
    return [float(line) for line in open(file_name)]

def avg(xs):
    return sum(xs) / len(xs)

def plot_speedup(process_times, file_name):
    all_times = process_times.values()
    single_process_average_time = avg(process_times[1])
    all_speedups = [[single_process_average_time / t for t in ts] for ts in all_times]

    plot(all_speedups, 'Speedup', file_name)


def plot_burndown(process_times, file_name):
    all_times = process_times.values()
    single_process_average_time = avg(process_times[1])
    all_burn_downs = [[single_process_average_time / t / processes for t in ts] for processes, ts in process_times.items()]

    plot(all_burn_downs, 'Burndown (Speedup / Processes)', file_name)


def plot(values, yaxis_label, file_name):
    figure = pyplot.figure(figsize=(20, 10))

    pyplot.xticks(range(1, len(values)))
    pyplot.boxplot(values)

    set_axes_on(figure)

    pyplot.ylim(ymin=0)

    pyplot.title('MatMul (SMP, 1000x1000)')
    pyplot.xlabel('MPI processes')
    pyplot.ylabel(yaxis_label)
    pyplot.show()

    pyplot.savefig(file_name, dpi=80)


def set_axes_on(figure):
    axes = figure.add_subplot(1,1,1)

    axes.xaxis.set_major_locator(MultipleLocator(1))
    axes.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes.xaxis.grid(True, 'major', linestyle='-', color='0.5', linewidth=1)
    axes.xaxis.grid(True, 'minor', linestyle='-', color='0.95', linewidth=1)

    axes.yaxis.set_major_locator(MultipleLocator(0.5))
    axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes.yaxis.set_minor_locator(MultipleLocator(0.25))
    axes.yaxis.grid(True, 'major', linestyle='-', color='0.5', linewidth=1)
    axes.yaxis.grid(True, 'minor', linestyle='-', color='0.95', linewidth=1)

    axes.set_axisbelow(True)



file_dir = sys.argv[1]
iterations = int(sys.argv[2]) if (len(sys.argv)>2) else 100
warmups = iterations / 10
matrix_size = 1000


def time_file(smpHosts, cudaHosts, ext = ".txt"):
    hostsString = ",".join(smpHosts) + "-" + ",".join(cudaHosts)
    return os.path.join(file_dir, "matmul_cluster_{0}_{1}{2}".format(hostsString, matrix_size, ext))

def command_line(smpHosts, cudaHosts):
    HOST_LINE =  "-host {0} -np 1 ./build/src/main/dwarf_mine -m {1} -c matrix -w {3} -n {4} --time_output {5} "\
                "--left_rows {2} --common_rows_columns {2} --right_columns {2}"

    time_file_name = time_file(smpHosts, cudaHosts)
    smp_lines = [HOST_LINE.format(host, "smp", matrix_size, warmups, iterations, time_file_name) for host in smpHosts]
    cuda_lines = [HOST_LINE.format(host, "cuda", matrix_size, warmups, iterations, time_file_name) for host in cudaHosts]

    mpirun_args = " : ".join(smp_lines + cuda_lines)

    return "mpirun --tag-output %s" % mpirun_args


def main():
    q1, q2, q3 = ("quadcore1", "quadcore2", "quadcore3")
    scenarios = [((q3,), ()), ((q3,q2), ()), ((q3,q2,q1), ())]

    numberOfScenarios = range(1, len(scenarios)+2)

    for smpHosts, cudaHosts in scenarios:
        print "Executing on smpHosts:", smpHosts, "and cudaHosts:", cudaHosts
        print "commandLine:", command_line(smpHosts, cudaHosts)
        startTime = datetime.now();
        os.system(command_line(smpHosts, cudaHosts))
        print "elapsed:", (datetime.now() - startTime).total_seconds(), "s"

    all_times = dict((len(smpHosts)+len(cudaHosts), times_from_file(time_file(smpHosts, cudaHosts))) for smpHosts, cudaHosts in scenarios)

    print "Plotting..."
    plot_speedup(all_times, time_file("", "", "_speedup.png"))
    plot_burndown(all_times, time_file("", "", "_burndown.png"))


main()