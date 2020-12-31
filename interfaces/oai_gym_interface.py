# Open AI Gym
import gym
import time


class OAIGymInterface(gym.Env):
    '''
    This is the Open AI gym interface class. The interface wraps the control path and ensures communication
    between the agent and the environment. The class descends from gym.Env, and is designed to be minimalistic (currently!).
    
    | **Args**
    | modules:                      Framework modules.
    | withGUI:                      If true, the module provides GUI control.
    | rewardCallback:               This callback function is invoked in the step routine in order to get the appropriate reward w.r.t. the experimental design.
    '''
    
    def __init__(self, modules, withGUI=True, rewardCallback=None):
        #PATRICK These are the variables for time at beginning and end of the trial
        self.timeAtStart = 0.0
        self.timeAtEnd = 0.0
        #PATRICK END
        # store the modules
        self.modules = modules    
        # store visual output variable
        self.withGUI = withGUI        
        # memorize the reward callback function
        self.rewardCallback = rewardCallback        
        self.world = self.modules['world']
        self.observations = self.modules['observation']        
        # second: action space
        self.action_space = modules['spatial_representation'].get_action_space()
        # third: observation space
        self.observation_space = modules['observation'].getObservationSpace()
        # all OAI spaces have been initialized!
        # this observation variable is filled by the OBS modules 
        self.observation = None
        # required for the analysis of the agent's behavior
        self.forbiddenZoneHit = False
        self.finalNode = -1
        # a variable that allows the OAI class to access the robotic agent class
        self.rlAgent = None
        
    def updateObservation(self,observation):
        '''
        This function updates the observation provided by the environment.
        
        | **Args**
        | observation:                  The new observation.
        '''
        self.observation = observation
    
    def step(self, action):
        '''
        The step function that propels the simulation.
        This function is called by the .fit function of the RL agent whenever a novel action has been computed.
        
        | **Args**
        | action:                       The action to be executed.
        '''
        callbackValue = self.modules['spatial_representation'].generate_behavior_from_action(action)
        callbackValue['rlAgent'] = self.rlAgent
        callbackValue['modules'] = self.modules
        reward, stopEpisode = self.rewardCallback(callbackValue)
        
        return self.modules['observation'].observation, reward, stopEpisode, {}
        
    def reset(self):
        '''
        This function restarts the RL agent's learning cycle by initiating a new episode.
        '''
        self.modules['spatial_representation'].generate_behavior_from_action('reset')
        
        return self.modules['observation'].observation
    
    #PATRICK
    def setTimeAtStart(self):
        '''
        This function sets the time at the start of the trial.
        '''
        self.timeAtStart = time.time()

    def setTimeAtEnd(self):
        '''
        This function sets the time at the end of the trial.
        '''
        self.timeAtEnd = time.time()
        
    def timeToComplete(self):
        '''
        This function calculates the time needed to complete a trial.
        '''
        elapsedTime = self.timeAtEnd - self.timeAtStart

        return elapsedTime
    #PATRICK END
