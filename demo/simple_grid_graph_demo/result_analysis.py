# This script os used to analyse the results of the experiments conducted in training_experiment.py
# Plots the graphs to the screen

import os
import numpy as np
import pyqtgraph as qg


def main():
    # Extracting data from files and saving it in a list of numpy arrays
    results = []
    trials = []
    memory = []
    for i in os.listdir("%s/results" % os.getcwd()):
        # Extracting metadata from file names
        numSessions = int(i[:1])
        numTrials = int(i[2:5])
        trials.append(numSessions * numTrials)
        memory.append(int(i[6:len(i) - 4]))

        # Reading data from file
        file = open(("%s\\results\\%s" % (os.getcwd(), i)), "r")
        dataStr = file.read()
        dataStr = dataStr[1:len(dataStr) - 1]
        results.append(np.fromstring(dataStr, sep=' '))

    print("Found %d result files" % len(results))

    # Plotting results in multiple graphs
    colors = ['r', 'b', 'g', 'y']
    for i in range(len(results)):
        qg.plot(np.arange(trials[i]), results[i], pen=colors[i],
                title='trials: %d, memory capacity: %d' % (trials[i], memory[i]))

    input("Press enter to continue")


if __name__ == "__main__":
    main()
