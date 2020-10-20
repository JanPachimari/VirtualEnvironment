# basic imports
import numpy     as np
# tensorflow/keras
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input
from tensorflow.keras.optimizers import Adam
# keras-rl
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

                    
class DQNAgentBaseline():
    '''
    The reinforcement learing class. It wraps all functionality required to set up a RL agent.
    
    | **Args**
    | guiParent:                    The widget that shows necessary visualization.
    | interfaceOAI:                 The interface to the Open AI Gym environment.
    | agentParams:                  The parameters of the agent to be used, provided as a dictionary.
    | visualOutput:                 If true, the module has to visualize the results.
    | maxEpochs:                    The maximum number of epochs to be logged.
    | memoryCapacity:               The capacity of the sequential memory used in the agent.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | trialBeginFcn:                The callback function called at the beginning of each trial, defined for more flexibility in scenario control.
    | trialEndFcn:                  The callback function called at the end of each trial, defined for more flexibility in scenario control.
    '''
     
    class callbacks(callbacks.Callback):
        '''
        The nested visualization class that is required by 'KERAS-RL' to visualize the training success (by means of episode reward)
        at the end of each episode, and update the policy visualization.
        
        | **Args**
        | rlParent:                     The RL-agent that hosts this class.
        | trialBeginFcn:                The callback function called in the beginning of each trial, defined for more flexibility in scenario control.
        | trialEndFcn:                  The callback function called at the end of each trial, defined for more flexibility in scenario control.
        '''
    
        def __init__(self, rlParent, trialBeginFcn=None, trialEndFcn=None):
            super(DQNAgentBaseline.callbacks, self).__init__()
            # store the hosting class
            self.rlParent = rlParent
            # store the trial end callback function
            self.trialBeginFcn = trialBeginFcn
            # store the trial end callback function
            self.trialEndFcn = trialEndFcn
            
        def on_episode_begin(self, epoch, logs):
            '''
            The following function is called whenever an epsisode startss, and updates the visual output in the plotted reward graphs.
            
            | **Args**
            | epoch:                        The current trial.
            | logs:                         Dictionary containing learning relevant statistics.
            '''
            # call custom callback function
            if self.trialBeginFcn is not None:
                self.trialBeginFcn(epoch, self.rlParent)
            
        def on_episode_end(self, epoch, logs):
            '''
            The following function is called whenever an episode ends, and updates the reward accumulator,
            simultaneously updating the visualization of the reward function.
            
            | **Args**
            | epoch:                        The current trial.
            | logs:                         Dictionary containing learning relevant statistics.
            '''
            if self.trialEndFcn is not None:
                self.trialEndFcn(epoch, self.rlParent, logs)
                
            
    def __init__(self, interfaceOAI, memoryCapacity=10000, epsilon=0.3, trialBeginFcn=None, trialEndFcn=None):
        # store the Open AI Gym interface
        self.interfaceOAI = interfaceOAI
        # prepare the model used in the reinforcement learner
        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.nb_actions = self.interfaceOAI.action_space.n
        # a sequential model is standardly used here, this model is subject to changes
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,) + self.interfaceOAI.observation_space.shape))
        self.model.add(Dense(units=64, activation='tanh'))
        self.model.add(Dense(units=64, activation='tanh'))
        self.model.add(Dense(units=64, activation='tanh'))
        self.model.add(Dense(units=64, activation='tanh'))
        self.model.add(Dense(units=self.nb_actions, activation='linear'))
        # prepare the memory for the RL agent
        self.memory = SequentialMemory(limit=memoryCapacity, window_length=1)
        # define the available policies
        policyEpsGreedy = EpsGreedyQPolicy(epsilon)
        # construct the agent
        # Retrieve the agent's parameters from the agentParams dictionary
        self.agent = DQNAgent(model=self.model, nb_actions=self.nb_actions, memory=self.memory, nb_steps_warmup=100,
                              enable_dueling_network=False, dueling_type='avg', target_model_update=1e-2, policy=policyEpsGreedy, batch_size=32)
        # compile the agent
        self.agent.compile(Adam(lr=.001,), metrics=['mse'])
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engagedCallbacks = self.callbacks(self, trialBeginFcn, trialEndFcn)
        
    def train(self, steps):
        '''
        This function is called to train the agent.
        
        | **Args**
        | steps:                        The maximum number of environmental steps.
        '''
        # call the fit method to start the RL learning process
        self.maxSteps = steps
        self.agent.fit(self.interfaceOAI, nb_steps=steps, verbose=0,callbacks=[self.engagedCallbacks],
                       nb_max_episode_steps=100, visualize=False)
        
    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.
        
        | **Args**
        | batch:                        The batch of states.
        '''
        return self.agent.model.predict_on_batch(batch)
        
