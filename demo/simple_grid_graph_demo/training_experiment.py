# This script is used to evaluate how fast the DQN agent can adapt to changing barrier configurations.
# It is part of the study project "Development of an Interface to Controll the Structure of a Virtual Environment" in winter term 2020/2021
# Based on simple_grid_graph_demo.py

# Remember to change the working directory and to give the world module your path to the blender executable!

# basic imports
from analysis.rl_monitoring.rl_performance_monitors import RLPerformanceMonitorBaseline
from interfaces.oai_gym_interface import OAIGymInterface
from observations.image_observations import ImageObservationBaseline
from agents.dqn_agents import DQNAgentBaseline
from spatial_representations.topology_graphs.manual_topology_graph_no_rotation import ManualTopologyGraphNoRotation
from frontends.frontends_blender import FrontendBlenderInterface
import os
import numpy as np
import pyqtgraph as qg
import random
# tensorflow/keras
from tensorflow.keras import backend
# change working dictionary
os.chdir("C:/Users/yoric/Desktop/virtual-environment-interface")
# import framework modules

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'!
visualOutput = True

# Defining experiment parameters
# Number of different barrier configurations
numSessions = 3
# Number of trials per session
numTrials = 100
# Memory capacity of the DQN agent
memoryCapacity = 5000

# The id's of the barriers to be rendered in each session
barrierConfigs = [[4, 5, 6, 7, 10], [16, 17, 18, 20, 21], [0, 5, 6, 8, 11]]


def rewardCallback(values):
    '''
    This is a callback function that defines the reward provided to the robotic agent.
    Note: this function has to be adopted to the current experimental design.

    | **Args**
    | values:                       A dict of values that are transferred from the OAI module to the reward function. This is flexible enough to accommodate for different experimental setups.
    '''
    # default reward
    reward = -1.0
    stopEpisode = False
    # node dependent reward
    if values['currentNode'].goalNode:
        reward = 10.0
        stopEpisode = True

    return [reward, stopEpisode]


def trialBeginCallback(trial, rlAgent):
    '''
    This is a callback function that is called in the beginning of each trial.
    Here, experimental behavior can be defined.

    | **Args**
    | trial:                        The number of the finished trial.
    | rlAgent:                      The employed reinforcement learning agent.

    '''
    print("Beginning trial", trial)
    if trial == rlAgent.trialNumber - 1:
        # end the experiment by setting the number of steps to a excessively large value, this stops the 'fit' routine
        rlAgent.agent.step = rlAgent.maxSteps + 1

    # Changing barrier configuration after set number of trials
    if trial % numTrials == 0:
        worldModule = rlAgent.interfaceOAI.modules['world']
        barriers = worldModule.get_barrierIDs()
        # Retrieving the barriers to be set from the global array
        set_barriers = barrierConfigs[trial//numTrials]
        for i in range(len(barriers)):
            worldModule.set_renderState(barriers[i], i in set_barriers)

        rlAgent.interfaceOAI.modules['spatial_representation'].reload()


def trialEndCallback(trial, rlAgent, logs):
    '''
    This is a callback routine that is called when a single trial ends.
    Here, functionality for performance evaluation can be introduced.

    | **Args**
    | trial:                        The number of the finished trial.
    | rlAgent:                      The employed reinforcement learning agent.
    | logs:                         Output of the reinforcement learning subsystem.

    '''
    print("Ending trial", trial)
    if visualOutput:
        # update the visual elements if required
        rlAgent.interfaceOAI.modules['spatial_representation'].updateVisualElements(
        )
        rlAgent.performanceMonitor.update(trial, logs)

    # Add reward to result
    rlAgent.interfaceOAI.modules['results'][trial] = logs['episode_reward']


def singleRun():
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''

    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    mainWindow = None
    # if visual output is required, activate an output window
    if visualOutput:
        mainWindow = qg.GraphicsWindow(title="workingTitle_Framework")
    # a dictionary that contains all employed modules
    modules = dict()
    modules['world'] = FrontendBlenderInterface(
        'simple_grid_graph_env/simple_grid_graph_maze.blend', 'C:/Users/yoric/blender-2.79b-windows64/blender')
    modules['observation'] = ImageObservationBaseline(
        modules['world'], mainWindow, visualOutput)
    modules['spatial_representation'] = ManualTopologyGraphNoRotation(
        modules, {'startNodes': [0], 'goalNodes': [15], 'cliqueSize': 4})
    modules['spatial_representation'].set_visual_debugging(
        visualOutput, mainWindow)
    modules['rl_interface'] = OAIGymInterface(
        modules, visualOutput, rewardCallback)
    modules['results'] = np.zeros(numTrials*numSessions)

    # initialize RL agent
    rlAgent = DQNAgentBaseline(
        modules['rl_interface'], memoryCapacity, 0.3, trialBeginCallback, trialEndCallback)
    # set the experimental parameters
    rlAgent.trialNumber = numTrials*numSessions
    # prepare performance monitor
    perfMon = RLPerformanceMonitorBaseline(rlAgent, mainWindow, visualOutput)
    rlAgent.performanceMonitor = perfMon
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rlAgent = rlAgent
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rlAgent = rlAgent
    # let the agent learn, with extremely large number of allowed maximum steps
    rlAgent.train(100000)
    # clear session
    backend.clear_session()
    # close blender

    modules['world'].stopBlender()
    # close GUI
    if visualOutput:
        mainWindow.close()

    return modules['results']


if __name__ == "__main__":
    # Conduct an appropriate ammount of experiments (Calls to singleRun) and calculate the mean of the results
    result = np.zeros((25, numTrials * numSessions))

    for i in range(25):
        result[i] = singleRun()

    finalResult = [np.mean(result[:, i]) for i in range(numTrials*numSessions)]
    qg.plot(np.arange(numTrials * numSessions), finalResult, pen='r')

    input("Press enter to continue")
