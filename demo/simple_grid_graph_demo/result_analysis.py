# This script os used to analyse the results of the experiments conducted in training_experiment.py
# Plots the graphs to the screen
# Contains solutions for either plotting all results in one window or indivudual windows
# Handles both .txt files containing only the final result(mean over multiple experiments) or .npy files containing results of all experiments seperately

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
        # Case 1: .txt file
        if i.endswith('.txt'):
            file = open(("%s\\results\\%s" % (os.getcwd(), i)), "r")
            dataStr = file.read()
            dataStr = dataStr[1:len(dataStr) - 1]
            results.append(np.fromstring(dataStr, sep=' '))
        # Case 2: .npy file
        else:
            filepath = "%s\\results\\%s" % (os.getcwd(), i)
            array = np.load(filepath)
            results.append(np.array([np.mean(array[:, i])
                                     for i in range(numTrials * numSessions)]))

    print("Found %d result files" % len(results))

    # Solution A:  Plotting results in multiple graphs
    colors = ['r', 'b', 'g', 'y']

    '''
    app = qg.QtGui.QApplication([])
    views = []

    for i in range(len(results)):

        view = qg.GraphicsView()
        l = qg.GraphicsLayout()
        view.setCentralItem(l)
        view.show()

        p = l.addPlot(row=(i//2), col=(i % 2),
                      title='trials: %d, memory capacity: %d' % (trials[i], memory[i]))
        p.plot(np.arange(trials[i]), results[i],  pen=colors[i])
        p.showGrid(x=True, y=True)

        views.append(view)
    '''

    # Solution B: Plotting results in one graph

    app = qg.QtGui.QApplication([])
    view = qg.GraphicsView()
    l = qg.GraphicsLayout()
    view.setCentralItem(l)
    view.show()

    for i in range(len(results)):
        p = l.addPlot(row=(i//2), col=(i % 2),
                      title='trials: %d, memory capacity: %d' % (trials[i], memory[i]))
        p.plot(np.arange(trials[i]), results[i],  pen=colors[i])
        p.showGrid(x=True, y=True)

    # Soulution C: Plotting in multiple graphs, no grid (deprecated)
    '''
    for i in range(len(results)):
        qg.plot(np.arange(trials[i]), results[i], pen=colors[i],
                title='trials: %d, memory capacity: %d' % (trials[i], memory[i]))
    '''

    input("Press enter to continue")


if __name__ == "__main__":
    main()
