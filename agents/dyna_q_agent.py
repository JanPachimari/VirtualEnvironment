# basic imports
import numpy as np
# memory module
from memory_modules.dyna_q_memory import DynaQMemory, GrowingDynaQMemory


class DynaQAgent():
    '''
    Implementation of a Dyna-Q agent.
    Q-function is represented as a static table.
    
    | **Args**
    | interfaceOAI:                 The interface to the Open AI Gym environment.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learningRate:                 The learning rate with which the Q-function is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    | trialEndFcn:                  The callback function called at the end of each trial, defined for more flexibility in scenario control.
    '''
    
    class callbacks():
        '''
        Callback class. Used for visualization and scenario control.
        
        | **Args**
        | rlParent:                     Reference to the Dyna-Q agent.
        | trialEndFcn:                  Maximum number of experiences that will be stored by the memory module.
        '''

        def __init__(self, rlParent, trialEndFcn=None):
            super(DynaQAgent.callbacks, self).__init__()
            # store the hosting class
            self.rlParent = rlParent
            # store the trial end callback function
            self.trialEndFcn = trialEndFcn
        
        def on_episode_end(self, epoch, logs):
            '''
            The following function is called whenever an episode ends, and updates the reward accumulator,
            simultaneously updating the visualization of the reward function.
            
            | **Args**
            | rlParent:                     Reference to the Dyna-Q agent.
            | trialEndFcn:                  Maximum number of experiences that will be stored by the memory module.
            '''
            if self.trialEndFcn is not None:
                self.trialEndFcn(epoch, self.rlParent, logs)
                
            
    def __init__(self, interfaceOAI, epsilon=0.3, beta=5, learningRate=0.9, gamma=0.99, trialEndFcn=None):
        # store the Open AI Gym interface
        self.interfaceOAI = interfaceOAI        
        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.numberOfActions = self.interfaceOAI.action_space.n
        # Q-learning parameters
        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma
        self.learningRate = learningRate
        # Q-table
        self.Q = np.zeros((self.interfaceOAI.world['states'], self.numberOfActions))
        # memory module
        self.M = DynaQMemory(self.interfaceOAI.world['states'], self.numberOfActions)
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engagedCallbacks = self.callbacks(self,trialEndFcn)
        
    def train(self, numberOfTrials=100, maxNumberOfSteps=50, replayBatchSize=100, noReplay=False):
        '''
        This function is called to train the agent.
        
        | **Args**
        | numberOfTrials:               The number of trials the Dyna-Q agent is trained.
        | maxNumberOfSteps:             The maximum number of steps per trial.
        | replayBatchSize:              The number of random that will be replayed.
        | noReplay:                     If true, experiences are not replayed.
        '''
        for trial in range(numberOfTrials):
            # reset environment
            state = self.interfaceOAI.reset()
            # log cumulative reward
            logs = {'episode_reward': 0}
            for step in range(maxNumberOfSteps):
                # determine next action
                qVals = self.retrieveQValues(state)
                action = self.selectAction(qVals, 'greedy', self.epsilon, self.beta)
                # perform action
                next_state, reward, stopEpisode, callbackValue = self.interfaceOAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - stopEpisode)}
                # store experience
                self.M.store(experience)
                # update Q-function with experience
                self.updateQ(experience)
                # update current state
                state = next_state
                # perform experience replay
                if not noReplay:
                    self.replay(replayBatchSize)
                # update cumulative reward
                logs['episode_reward'] += reward
                # stop trial when the terminal state is reached
                if stopEpisode:
                    break
            # callback
            self.engagedCallbacks.on_episode_end(trial, logs)
    
    def updateQ(self, experience):
        '''
        This function updates the Q-function with a given experience.
        
        | **Args**
        | experience:                   The experience with which the Q-function will be updated.
        '''
        # compute TD-error
        td = experience['reward']
        td += self.gamma * experience['terminal'] * np.amax(self.retrieveQValues(experience['next_state']))
        td -= self.retrieveQValues(experience['state'])[experience['action']]
        # update Q-function with TD-error
        self.Q[experience['state']][experience['action']] += self.learningRate * td
            
    def replay(self, replayBatchSize=200):
        '''
        This function replays experiences to update the Q-function.
        
        | **Args**
        | replayBatchSize:              The number of random that will be replayed.
        '''
        # sample random batch of experiences
        replayBatch = self.M.retrieveRandom(replayBatchSize)
        # update the Q-function with each experience
        for experience in replayBatch:
            self.updateQ(experience)
    
    def retrieveQValues(self, state):
        '''
        This function retrieves Q-values for a given state.
        
        | **Args**
        | state:                        The state for which Q-values should be retrieved.
        '''
        # retrieve Q-values, if entry exists
        return self.Q[state]

    def selectAction(self, qVals, method='greedy', epsilon=0.3, beta=5):
        '''
        This function selects an action according to the Q-values.
        
        | **Args**
        | qVals:                        The Q-values.
        | method:                       The method used for action selection.
        | epsilon:                      The epsilon parameter used under greedy action selection.
        | beta:                         The beta parameter used when applying the softmax function to the Q-values.
        '''
        # revert to 'argmax' in case that the method name is not valid
        if not method in ['greedy', 'softmax']:
            method = 'greedy'
        # select action with highest value
        if method == 'greedy':
            # in case that Q-values are equal select a random action
            if np.all(qVals == qVals[0]) or np.random.rand() < epsilon:
                return np.random.randint(qVals.shape[0])
            return np.argmax(qVals)
        # select action probabilistically
        elif method == 'softmax':
            probs = np.exp(beta * qVals)/np.sum(np.exp(beta * qVals))
            return np.random.choice(qVals.shape[0], p=probs)
    
    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.
        
        | **Args**
        | batch:                        The batch of states.
        '''
        predictions = []
        for state in batch:
            predictions += [self.retrieveQValues(state)]
            
        return np.array(predictions)


class GrowingDynaQAgent():
    '''
    Implementation of a Dyna-Q agent.
    Q-function is represented as a growing table.
    
    | **Args**
    | interfaceOAI:                 The interface to the Open AI Gym environment.
    | memoryCapacity:               Maximum number of experiences that will be stored by the memory module.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learningRate:                 The learning rate with which the Q-function is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    | trialEndFcn:                  The callback function called at the end of each trial, defined for more flexibility in scenario control.
    | randomProjection:             If true, observations are randomly projected to a lower dimensional vector.
    | randomFeatures:               The number of random features.
    '''
    
    class callbacks():
        '''
        Callback class. Used for visualization and scenario control.
        
        | **Args**
        | rlParent:                     Reference to the Dyna-Q agent.
        | trialEndFcn:                  Maximum number of experiences that will be stored by the memory module.
        '''

        def __init__(self, rlParent, trialEndFcn=None):
            super(GrowingDynaQAgent.callbacks, self).__init__()
            # store the hosting class
            self.rlParent = rlParent
            # store the trial end callback function
            self.trialEndFcn = trialEndFcn
        
        def on_episode_end(self, epoch, logs):
            '''
            The following function is called whenever an episode ends, and updates the reward accumulator,
            simultaneously updating the visualization of the reward function.
            
            | **Args**
            | rlParent:                     Reference to the Dyna-Q agent.
            | trialEndFcn:                  Maximum number of experiences that will be stored by the memory module.
            '''
            if self.trialEndFcn is not None:
                self.trialEndFcn(epoch, self.rlParent, logs)
                
            
    def __init__(self, interfaceOAI, memoryCapacity=10000, epsilon=0.3, beta=5, learningRate=0.9, gamma=0.99, trialEndFcn=None, randomProjection=False, randomFeatures=64):
        # store the Open AI Gym interface
        self.interfaceOAI = interfaceOAI        
        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.numberOfActions = self.interfaceOAI.action_space.n
        # Q-learning parameters
        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma
        self.learningRate = learningRate
        # Q-table
        self.Q = dict()
        # memory module
        self.M = GrowingDynaQMemory()
        # random projection Matrix
        self.randomProjection = randomProjection
        if self.randomProjection:
            self.P = np.random.rand(randomFeatures, np.product(self.interfaceOAI.observation_space.shape))
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engagedCallbacks = self.callbacks(self,trialEndFcn)
        
    def train(self, numberOfTrials=100, maxNumberOfSteps=50, replayBatchSize=100, noReplay=False):
        '''
        This function is called to train the agent.
        
        | **Args**
        | numberOfTrials:               The number of trials the Dyna-Q agent is trained.
        | maxNumberOfSteps:             The maximum number of steps per trial.
        | replayBatchSize:              The number of random that will be replayed.
        | noReplay:                     If true, experiences are not replayed.
        '''
        for trial in range(numberOfTrials):
            # reset environment
            observation = self.interfaceOAI.reset()
            # log cumulative reward
            logs = {'episode_reward': 0}
            for step in range(maxNumberOfSteps):
                # make state tuple
                obs = np.copy(observation.flatten())
                if self.randomProjection:
                    obs = np.inner(self.P, obs)
                state = tuple(obs.flatten())
                # determine next action
                qVals = self.retrieveQValues(state)
                action = self.selectAction(qVals, 'greedy', self.epsilon, self.beta)
                # perform action
                observation, reward, stopEpisode, callbackValue = self.interfaceOAI.step(action)
                # make state tuple
                obs = np.copy(observation.flatten())
                if self.randomProjection:
                    obs = np.inner(self.P, obs)
                next_state = tuple(obs.flatten())
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - stopEpisode)}
                # store experience
                self.M.store(experience)
                # update Q-function with experience
                self.updateQ(experience)
                # perform experience replay
                if not noReplay:
                    self.replay(replayBatchSize)
                # update cumulative reward
                logs['episode_reward'] += reward
                # stop trial when the terminal state is reached
                if stopEpisode:
                    break
            # callback
            self.engagedCallbacks.on_episode_end(trial, logs)
    
    def updateQ(self, experience):
        '''
        This function updates the Q-function with a given experience.
        
        | **Args**
        | experience:                   The experience with which the Q-function will be updated.
        '''
        # add new entry to the Q-function if necessary (assume 0 reward for all actions)
        if not experience['state'] in self.Q:
            self.Q[experience['state']] = np.zeros(self.numberOfActions)
        # compute TD-error
        td = experience['reward']
        td += self.gamma * experience['terminal'] * np.amax(self.retrieveQValues(experience['next_state']))
        td -= self.retrieveQValues(experience['state'])[experience['action']]
        # update Q-function with TD-error
        self.Q[experience['state']][experience['action']] += self.learningRate * td
            
    def replay(self, replayBatchSize=200):
        '''
        This function replays experiences to update the Q-function.
        
        | **Args**
        | replayBatchSize:              The number of random that will be replayed.
        '''
        # sample random batch of experiences
        replayBatch = self.M.retrieveRandom(replayBatchSize)
        # update the Q-function with each experience
        for experience in replayBatch:
            self.updateQ(experience)
    
    def retrieveQValues(self, state):
        '''
        This function retrieves Q-values for a given state.
        
        | **Args**
        | state:                        The state for which Q-values should be retrieved.
        '''
        # retrieve Q-values, if entry exists
        if state in self.Q:
            return self.Q[state]
        # else, return 0 valued actions
        else:
            return np.zeros(self.numberOfActions)

    def selectAction(self, qVals, method='greedy', epsilon=0.3, beta=5):
        '''
        This function selects an action according to the Q-values.
        
        | **Args**
        | qVals:                        The Q-values.
        | method:                       The method used for action selection.
        | epsilon:                      The epsilon parameter used under greedy action selection.
        | beta:                         The beta parameter used when applying the softmax function to the Q-values.
        '''
        # revert to 'argmax' in case that the method name is not valid
        if not method in ['greedy', 'softmax']:
            method = 'greedy'
        # select action with highest value
        if method == 'greedy':
            # in case that Q-values are equal select a random action
            if np.all(qVals == qVals[0]) or np.random.rand() < epsilon:
                return np.random.randint(qVals.shape[0])
            return np.argmax(qVals)
        # select action probabilistically
        elif method == 'softmax':
            probs = np.exp(beta * qVals)/np.sum(np.exp(beta * qVals))
            return np.random.choice(qVals.shape[0], p=probs)
    
    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.
        
        | **Args**
        | batch:                        The batch of states.
        '''
        predictions = []
        for experience in batch:
            # make state tuple
            state = tuple(experience.flatten())
            # retrieve Q-values
            predictions += [self.retrieveQValues(state)]
            
        return np.array(predictions) 