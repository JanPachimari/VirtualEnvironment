# basic imports
import numpy as np


class DynaQMemory():
    '''
    Memory module to be used with the Dyna-Q agent.
    Experiences are stored as a static table.
    
    | **Args**
    | numberOfStates:               The number of environmental states.
    | numberOfActions:              The number of the agent's actions.
    | learningRate:                 The learning rate with which experiences are updated.
    '''
    
    def __init__(self, numberOfStates, numberOfActions, learningRate=0.9):
        # initialize variables
        self.learningRate = learningRate
        self.rewards = np.zeros((numberOfStates, numberOfActions))
        self.states = np.zeros((numberOfStates, numberOfActions)).astype(int)
        self.terminals = np.zeros((numberOfStates, numberOfActions)).astype(int)
        
    def store(self, experience):
        '''
        This function stores a given experience.
        
        | **Args**
        | experience:                   The experience to be stored.
        '''
        # update experience
        self.rewards[experience['state']][experience['action']] += self.learningRate * (experience['reward'] - self.rewards[experience['state']][experience['action']])
        self.states[experience['state']][experience['action']] = experience['next_state']
        self.terminals[experience['state']][experience['action']] = experience['terminal']
            
    def retrieve(self, state, action):
        '''
        This function retrieves a specific experience.
        
        | **Args**
        | state:                        The environmental state.
        | action:                       The action selected.
        '''
        return {'state': state, 'action': action, 'reward': self.rewards[state][action],
                'next_state': self.states[state][action], 'terminal': self.terminals[state][action]}
        
    def retrieveRandom(self, numberOfExperiences=1):
        '''
        This function retrieves a number of random experiences.
        
        | **Args**
        | numberOfExperiences:          The number of random experiences to be drawn.
        '''
        # draw random experiences
        experiences = []
        for exp in range(numberOfExperiences):
            state = np.random.randint(self.states.shape[0])
            action = np.random.randint(self.states.shape[1]) 
            experiences += [{'state': state, 'action': action, 'reward': self.rewards[state][action],
                             'next_state': self.states[state][action], 'terminal': self.terminals[state][action]}]
            
        return experiences


class GrowingDynaQMemory():
    '''
    Memory module to be used with the Dyna-Q agent.
    Experiences are stored as a growing table.
    
    | **Args**
    | memoryCapacity:               Maximum number of experiences that will be stored.
    | learningRate:                 The learning rate with which experiences are updated.
    '''
    
    def __init__(self, memoryCapacity=1000, learningRate=0.9):
        # initialize variables
        self.memoryCapacity = memoryCapacity
        self.learningRate = learningRate
        self.memory = dict()
        
        
    def store(self, experience):
        '''
        This function stores a given experience.
        
        | **Args**
        | experience:                   The experience to be stored.
        '''
        # make state-action key
        state_action = experience['state'] + tuple([experience['action']])
        # if entry already exists, update it with the experience
        if state_action in self.memory:
            self.memory[state_action]['reward'] += self.learningRate * (experience['reward'] - self.memory[state_action]['reward'])
            self.memory[state_action]['next_state'] = experience['next_state']
        # else, just make a new entry from the experience
        else:
            self.memory[state_action] = experience
        # enforce capacity limit
        if len(self.memory) > self.memoryCapacity:
            self.memory.pop(list(self.memory)[0])
            
    def retrieve(self, state, action):
        '''
        This function retrieves a specific experience.
        
        | **Args**
        | state:                        The environmental state.
        | action:                       The action selected.
        '''
        # make state-action key
        state_action = tuple(state) + tuple(action)
        # if it exists, retrieve experience
        if state_action in self.memory:
            return self.memory[state_action]
        # else return nothing
        else:
            return None
        
    def retrieveRandom(self, numberOfExperiences=1):
        '''
        This function retrieves a number of random experiences.
        
        | **Args**
        | numberOfExperiences:          The number of random experiences to be drawn.
        '''
        # draw random experiences
        experiences = []
        for exp in range(numberOfExperiences):
            idx = np.random.randint(len(self.memory))
            experiences += [self.memory[list(self.memory)[idx]]]
            
        return experiences
    
    
class DynaQSequentialReplayMemory():
    '''
    Memory module to be used with the sequDyna-Q agent.
    Experiences are stored as a static table.
    
    | **Args**
    | SST:                          The state-state transition probabilities.
    | numberOfActions:              The number of the agent's actions.
    | gamma:                        The discount factor used to compute the successor representation.
    | decay_inhibition:             The factor by which inhibition is decayed.
    | decay_strength_slow:          The factor by which the slow decaying experience strengths are decayed.
    | decay_strength_fast:          The factor by which the fast decaying experience strengths are decayed.
    | learningRate:                 The learning rate with which experiences are updated.
    | simpleStrength:               If true, only the slow decaying experience strengths are used during replay.
    '''
    
    def __init__(self, SST, numberOfActions, gamma, decay_inhibition, decay_strength_slow, decay_strength_fast, learningRate=0.9, simpleStrength=True):
        # initialize variables
        self.numberOfStates = SST.shape[0]
        self.numberOfActions = numberOfActions
        self.decay_inhibition = decay_inhibition
        self.decay_strength_slow = decay_strength_slow
        self.decay_strength_fast = decay_strength_fast
        self.learningRate = learningRate
        self.simpleStrength = simpleStrength
        # compute successor representation
        self.SR = np.linalg.inv(np.eye(SST.shape[0]) - gamma*SST)
        #self.SR = rescale(self.SR)
        # prepare memory structures
        self.rewards = np.zeros((self.numberOfStates, self.numberOfActions))
        self.states = np.zeros((self.numberOfStates, self.numberOfActions)).astype(int)
        self.terminals = np.zeros((self.numberOfStates, self.numberOfActions)).astype(int)
        # prepare replay-relevant structures
        self.C_slow = np.zeros(self.numberOfStates * self.numberOfActions)
        self.C_fast = np.zeros(self.numberOfStates * self.numberOfActions)
        self.I = np.zeros(self.numberOfStates)
        
    def store(self, experience):
        '''
        This function stores a given experience.
        
        | **Args**
        | experience:                   The experience to be stored.
        '''
        state, action = experience['state'], experience['action']
        # update experience
        self.rewards[state][action] += self.learningRate * (experience['reward'] - self.rewards[state][action])
        self.states[state][action] = experience['next_state']
        self.terminals[state][action] = experience['terminal']
        # update replay-relevent structures
        self.C_slow *= self.decay_strength_slow
        self.C_slow[self.numberOfStates * action + state] += 1
        self.C_fast *= self.decay_strength_fast
        self.C_fast[self.numberOfStates * action + state] += 10
    
    def replay(self, replayLength, current_state=None):
        '''
        This function replays experiences.
        
        | **Args**
        | replayLength:                 The number of experiences that will be replayed.
        | current_state:                State at which replay should start.
        '''
        # if a state is not defined, then choose the one with highest experience strength
        if current_state is None:
            current_state = np.argmax(self.C_slow + self.C_fast * (1 - self.simpleStrength))
        # reset inhibition
        self.I *= 0
        # replay
        experiences = []
        for step in range(replayLength):
            # compute activation ratings
            #C = (self.C_slow + self.C_fast * (1 - self.simpleStrength))
            C = np.copy(self.C_slow)
            C /= np.amax(C)
            D = np.tile(self.SR[current_state], self.numberOfActions)
            I = np.tile(self.I, self.numberOfActions)
            R = C * D * (1 - I)
            # compute activation probabilities
            probs = softmax(R, -1)
            probs = probs/np.sum(probs)
            # determine state and action
            exp = np.random.choice(np.arange(0,probs.shape[0]), p=probs)
            action = int(exp/self.numberOfStates)
            current_state = exp - (action * self.numberOfStates)
            # inhibit state
            self.I *= self.decay_inhibition
            self.I[current_state] = 1
            # "reactivate" experience
            experience = {'state': current_state, 'action': action, 'reward': self.rewards[current_state][action],
                          'next_state': self.states[current_state][action], 'terminal': self.terminals[current_state][action]}
            experiences += [experience]
            
        return experiences
    
def softmax(data, offset=0, beta=5):
    '''
    This function computes the softmax over the input.
    
    | **Args**
    | data:                         Input of the softmax function.
    | beta:                         Beta value.
    '''
    exp = np.exp(data * beta) + offset
    
    return exp/np.sum(exp)


def rescale(SRMatrix):
    '''
    This function normalizes each row of the Successor Representation matrix so that 1 is the maximum value.
    
    | **Args**
    | SRMatrix:                     Input SR matrix.
    '''
    scaledSRMatrix = np.copy(SRMatrix)
    for r, row in enumerate(scaledSRMatrix):
        scaledSRMatrix[r] = row/np.amax(row)
    
    return scaledSRMatrix


class DynaQSequentialReplayMemoryDist():
    '''
    Memory module to be used with the sequDyna-Q agent.
    Experiences are stored as a static table.
    
    | **Args**
    | coordinates:                  The coordinates of all environmental states.
    | numberOfActions:              The number of the agent's actions.
    | gamma:                        The discount factor used to compute the successor representation.
    | decay_inhibition:             The factor by which inhibition is decayed.
    | decay_strength_slow:          The factor by which the slow decaying experience strengths are decayed.
    | decay_strength_fast:          The factor by which the fast decaying experience strengths are decayed.
    | learningRate:                 The learning rate with which experiences are updated.
    | simpleStrength:               If true, only the slow decaying experience strengths are used during replay.
    '''
    
    def __init__(self, coordinates, numberOfActions, gamma, decay_inhibition, decay_strength_slow, decay_strength_fast, learningRate=0.9, simpleStrength=True):
        # initialize variables
        self.numberOfStates = coordinates.shape[0]
        self.numberOfActions = numberOfActions
        self.decay_inhibition = decay_inhibition
        self.decay_strength_slow = decay_strength_slow
        self.decay_strength_fast = decay_strength_fast
        self.learningRate = learningRate
        self.simpleStrength = simpleStrength
        self.coordinates = coordinates
        # prepare memory structures
        self.rewards = np.zeros((self.numberOfStates, self.numberOfActions))
        self.states = np.zeros((self.numberOfStates, self.numberOfActions)).astype(int)
        self.terminals = np.zeros((self.numberOfStates, self.numberOfActions)).astype(int)
        # prepare replay-relevant structures
        self.C_slow = np.zeros(self.numberOfStates * self.numberOfActions)
        self.C_fast = np.zeros(self.numberOfStates * self.numberOfActions)
        self.I = np.zeros(self.numberOfStates)
        
    def store(self, experience):
        '''
        This function stores a given experience.
        
        | **Args**
        | experience:                   The experience to be stored.
        '''
        state, action = experience['state'], experience['action']
        # update experience
        self.rewards[state][action] += self.learningRate * (experience['reward'] - self.rewards[state][action])
        self.states[state][action] = experience['next_state']
        self.terminals[state][action] = experience['terminal']
        # update replay-relevent structures
        self.C_slow *= self.decay_strength_slow
        self.C_slow[self.numberOfStates * action + state] += 1
        self.C_fast *= self.decay_strength_fast
        self.C_fast[self.numberOfStates * action + state] += 10
    
    def replay(self, replayLength, current_state=None):
        '''
        This function replays experiences.
        
        | **Args**
        | replayLength:                 The number of experiences that will be replayed.
        | current_state:                State at which replay should start.
        '''
        # if a state is not defined, then choose the one with highest experience strength
        if current_state is None:
            current_state = np.argmax(self.C_slow + self.C_fast * (1 - self.simpleStrength))
        # reset inhibition
        self.I *= 0
        # replay
        experiences = []
        for step in range(replayLength):
            # compute activation ratings
            #C = (self.C_slow + self.C_fast * (1 - self.simpleStrength))
            C = np.copy(self.C_slow)
            C /= np.amax(C)
            D = np.tile(self.computeDists(current_state), self.numberOfActions)
            I = np.tile(self.I, self.numberOfActions)
            R = C * D * (1 - I)
            # compute activation probabilities
            probs = softmax(R, -1, 500)
            probs = probs/np.sum(probs)
            # determine state and action
            exp = np.random.choice(np.arange(0,probs.shape[0]), p=probs)
            action = int(exp/self.numberOfStates)
            current_state = exp - (action * self.numberOfStates)
            # inhibit state
            self.I *= self.decay_inhibition
            self.I[current_state] = 1
            # "reactivate" experience
            experience = {'state': current_state, 'action': action, 'reward': self.rewards[current_state][action],
                          'next_state': self.states[current_state][action], 'terminal': self.terminals[current_state][action]}
            experiences += [experience]
            
        return experiences
    
    def computeDists(self, state):
        dists = np.sqrt(np.sum((self.coordinates - self.coordinates[state])**2, axis=1))
        
        return np.exp(-dists)
    
def softmax(data, offset=0, beta=5):
    '''
    This function computes the softmax over the input.
    
    | **Args**
    | data:                         Input of the softmax function.
    | beta:                         Beta value.
    '''
    exp = np.exp(data * beta) + offset
    
    return exp/np.sum(exp)


class PMAMemory():
    '''
    Memory module to be used with the Dyna-Q agent.
    Experiences are stored as a static table.
    
    | **Args**
    | numberOfStates:               The number of environmental states.
    | numberOfActions:              The number of the agent's actions.
    | learningRate:                 The learning rate with which experiences are updated.
    '''
    
    def __init__(self, rlAgent, numberOfStates, numberOfActions, learningRate=0.9):
        # initialize variables
        self.rlAgent = rlAgent
        self.numberOfStates = numberOfStates
        self.numberOfActions = numberOfActions
        self.learningRate = learningRate
        self.learningRateSR = 0.9
        self.gammaSR = 0.9
        self.SST = np.sum(self.rlAgent.interfaceOAI.world['sas'], axis=1)/self.numberOfActions
        self.rewards = np.zeros((numberOfStates, numberOfActions))
        self.states = np.zeros((numberOfStates, numberOfActions)).astype(int)
        self.terminals = np.zeros((numberOfStates, numberOfActions)).astype(int)
        
    def store(self, experience):
        '''
        This function stores a given experience.
        
        | **Args**
        | experience:                   The experience to be stored.
        '''
        # update experience
        self.rewards[experience['state']][experience['action']] += self.learningRate * (experience['reward'] - self.rewards[experience['state']][experience['action']])
        self.states[experience['state']][experience['action']] = experience['next_state']
        self.terminals[experience['state']][experience['action']] = experience['terminal']
        # update memory structures
        targetVector = np.zeros(self.numberOfStates)
        targetVector[experience['next_state']] = 1
        self.SST[experience['state']] += self.learningRateSR * (targetVector - self.SST[experience['state']])
        
    def replay(self, numberOfSteps=20, state=None):
        SR = np.linalg.inv(np.eye(self.SST.shape[0]) - self.gammaSR*self.SST)[state]
        for step in range(numberOfSteps):
            utility = self.computeGain() * np.tile(SR, self.numberOfActions)
            maxUtil = np.argmax(utility)
            bestAction = int(maxUtil/self.numberOfStates)
            bestState = maxUtil - self.numberOfStates * bestAction
            exp = self.retrieve(bestState, bestAction)
            self.rlAgent.updateQ(exp)
        
    def computeGain(self):
        gain = np.zeros(self.numberOfStates * self.numberOfActions)
        for state in range(self.numberOfStates):
            for action in range(self.numberOfActions):
                exp = self.retrieve(state, action)
                Qs = self.rlAgent.Q[state]
                QsNew = np.copy(Qs)
                # compute TD-error
                td = exp['reward']
                td += self.rlAgent.gamma * exp['terminal'] * np.amax(self.rlAgent.retrieveQValues(exp['next_state']))
                td -= self.rlAgent.retrieveQValues(exp['state'])[exp['action']]
                QsNew[action] += self.rlAgent.learningRate * td
                gain[self.numberOfStates * action + state] = np.amax(QsNew) - np.amax(Qs)
        gain = np.clip(gain, a_min=10**-9, a_max=None)
        
        return gain
            
    def retrieve(self, state, action):
        '''
        This function retrieves a specific experience.
        
        | **Args**
        | state:                        The environmental state.
        | action:                       The action selected.
        '''
        return {'state': state, 'action': action, 'reward': self.rewards[state][action],
                'next_state': self.states[state][action], 'terminal': self.terminals[state][action]}
        
    def retrieveRandom(self, numberOfExperiences=1):
        '''
        This function retrieves a number of random experiences.
        
        | **Args**
        | numberOfExperiences:          The number of random experiences to be drawn.
        '''
        # draw random experiences
        experiences = []
        for exp in range(numberOfExperiences):
            state = np.random.randint(self.states.shape[0])
            action = np.random.randint(self.states.shape[1]) 
            experiences += [{'state': state, 'action': action, 'reward': self.rewards[state][action],
                             'next_state': self.states[state][action], 'terminal': self.terminals[state][action]}]
            
        return experiences