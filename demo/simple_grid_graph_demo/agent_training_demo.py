#Remember to change the working directory and to give the world module your path to the blender executable!

# basic imports
import os
import numpy as np
import pyqtgraph as qg
import random
import time
# tensorflow/keras
from tensorflow.keras import backend
# change working dictionary
#os.chdir("")
# import framework modules
from frontends.frontends_blender import FrontendBlenderInterface
from spatial_representations.topology_graphs.manual_topology_graph_no_rotation import ManualTopologyGraphNoRotation
from agents.dqn_agents import DQNAgentBaseline
from observations.image_observations import ImageObservationBaseline
from interfaces.oai_gym_interface import OAIGymInterface
from analysis.rl_monitoring.rl_performance_monitors import RLPerformanceMonitorBaseline

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'!
visualOutput = True

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
    rlAgent.interfaceOAI.setTimeAtStart()
    print("Beginning trial", trial)
    
    if trial == rlAgent.trialNumber - 1:
        # end the experiment by setting the number of steps to a excessively large value, this stops the 'fit' routine
        rlAgent.agent.step = rlAgent.maxSteps + 1

def trialEndCallback(trial,rlAgent,logs):
    '''
    This is a callback routine that is called when a single trial ends.
    Here, functionality for performance evaluation can be introduced.
    
    | **Args**
    | trial:                        The number of the finished trial.
    | rlAgent:                      The employed reinforcement learning agent.
    | logs:                         Output of the reinforcement learning subsystem.

    '''
    #This takes the difference between the time at the start and at the end of the trial to calculate ho mucht time it took the agent to complete a trial.
    #This time should steadily go down as the training process continues and go up once there is a change in the environment.
    rlAgent.interfaceOAI.setTimeAtEnd()
    elapsedTime = rlAgent.interfaceOAI.timeToComplete()
    print("Ending trial", trial, "in about", elapsedTime, "seconds")
    #PATRICK This sets 2 barriers after a set number of trials to see how the agent behaves after a change in the environment.
    
    if trial == 99:
        rlAgent.interfaceOAI.modules['world'].set_barrier('barrier013-014', 'True', 0, '//textures/wall_01.bmp')
        rlAgent.interfaceOAI.modules['world'].set_barrier('barrier007-011', 'True', 0, '//textures/wall_01.bmp')

        rlAgent.interfaceOAI.modules['spatial_representation'].reload()
        print("Changed configuration at trial", trial)
    #PATRICK END




    if visualOutput:
        # update the visual elements if required
        rlAgent.interfaceOAI.modules['spatial_representation'].updateVisualElements()
        rlAgent.performanceMonitor.update(trial, logs)


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
        mainWindow = qg.GraphicsWindow( title="workingTitle_Framework" )
    # a dictionary that contains all employed modules
    modules=dict()
    modules['world']=FrontendBlenderInterface('simple_grid_graph_env/simple_grid_graph_maze.blend', 'BLENDER_EXECUTABLE_PATH')
    modules['observation']=ImageObservationBaseline(modules['world'],mainWindow,visualOutput)
    modules['spatial_representation']=ManualTopologyGraphNoRotation(modules,{'startNodes':[0],'goalNodes':[15],'cliqueSize':4})
    modules['spatial_representation'].set_visual_debugging(visualOutput,mainWindow)
    modules['rl_interface']=OAIGymInterface(modules,visualOutput,rewardCallback)

    # initialize RL agent
    rlAgent = DQNAgentBaseline(modules['rl_interface'], 5000, 0.3, trialBeginCallback, trialEndCallback)
    # set the experimental parameters
    rlAgent.trialNumber = 500
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

if __name__ == "__main__":    
    singleRun()
