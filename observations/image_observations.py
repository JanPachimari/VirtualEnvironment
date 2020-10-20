# basic imports
import numpy as np
import math
# Open CV
import cv2
# Qt
from PyQt5.QtCore import QRectF
# Ai Gym
import gym
from gym import spaces
# pyqt graph
import pyqtgraph as qg


class ImageObservationBaseline():
    '''
    This module computes an observation based on the current camera image acquired by the robot/agent.
    
    | **Args**
    | world:                        The world module.
    | guiParent:                    The main window for visualization.
    | visualOutput:                 If true, the module provides visual output.
    '''

    def __init__(self, world, guiParent, visualOutput=True):
        # store the world module reference
        self.worldModule = world
        self.topologyModule = None
        self.visualOutput = visualOutput
        self.observation = None
        # generate a visual display of the observation
        if self.visualOutput:
                # add the graph plot to the GUI widget
                self.plot = guiParent.addPlot(title='Camera image observation')
                # set extension of the plot, lock aspect ratio
                self.plot.setAspectLocked()                
                # add the camera image plot item
                self.cameraImage = qg.ImageItem()                
                self.plot.addItem(self.cameraImage)
                # add the observation plot item
                self.observationImage = qg.ImageItem()
                self.plot.addItem(self.observationImage)
        # a list of reference images captured in the preparation phase.
        # those reference images will drive the observationFromPose function
        self.imageDims = (30, 1)
    
    def update(self):
        '''
        This function updates the current observation.
        '''     
        # the observation is plainly the robot's camera image data
        observation = self.worldModule.envData['imageData']
        
        # display the observation camera image
        if self.visualOutput:
            imageData = observation
            self.cameraImage.setOpts(axisOrder='row-major')
            imageData = imageData[:, :, ::-1]
            self.cameraImage.setImage(imageData)
            imageScale = 1.
            self.cameraImage.setRect(QRectF(0.0, 0.0, imageScale, imageData.shape[0]/imageData.shape[1]*imageScale))
    
        # scale the one-line image to further reduce computational demands
        observation = cv2.resize(observation, dsize=self.imageDims)
        observation.astype('float32')
        observation = observation/255.0
        # display the observation camera image reduced to one line
        if self.visualOutput:
            imageData = observation
            self.observationImage.setOpts(axisOrder='row-major')
            imageData = imageData[:, :, ::-1]
            self.observationImage.setImage(imageData)
            imageScale = 1
            self.observationImage.setRect(QRectF(0.0, -0.1, imageScale, imageData.shape[0]/imageData.shape[1]*imageScale))
        
        self.observation = observation
        
    def getObservationSpace(self):
        '''
        This function returns the observation space for the given observation class.
        ''' 
        # currently, use a one-line 'image' to save computational resources
        observation_space = gym.spaces.Box (low=0.0, high=1.0 ,shape=(self.imageDims[1], self.imageDims[0], 3))
        return observation_space