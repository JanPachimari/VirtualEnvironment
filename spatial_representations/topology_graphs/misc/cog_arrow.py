# basic imports
import numpy as np
# Qt
import PyQt5 as qt
import pyqtgraph as qg
from PyQt5 import QtGui


class CogArrow(qg.ArrowItem):
    '''
    Helper class for the visualization of the topology graph.
    Constructs a centered arrow pointing in a dedicated direction, inherits from 'ArrowItem'.
    '''
    
    def setData(self, x, y, angle):
        '''
        This function sets the arrow's position and orientation.
        
        | **Args**
        | x:                            The x coordinate of the arrow's center.
        | y:                            The y coordinate of the arrow's center.
        | angle:                        The arrow's orientation.
        '''
        # the angle has to be modified to suit the demands of the environment(?)
        angle = -angle/np.pi * 180.0 + 180.0
        # assemble a new temporary dict that is used for path construction
        tempOpts = dict()
        tempOpts['headLen'] = self.opts['headLen']
        tempOpts['tipAngle'] = self.opts['tipAngle']
        tempOpts['baseAngle'] = self.opts['baseAngle']
        tempOpts['tailLen'] = self.opts['tailLen']
        tempOpts['tailWidth'] = self.opts['tailWidth']
        # create the path
        arrowPath = qg.functions.makeArrowPath(**tempOpts)
        # identify boundaries of the arrows, required to shif the arrow
        bounds = arrowPath.boundingRect()
        # prepare a transform
        transform = QtGui.QTransform()
        # shift and rotate the path (arrow)
        transform.rotate(angle)
        transform.translate(int(-float(bounds.x())-float(bounds.width())/10.0*7.0), int(float(-bounds.y())-float(bounds.height())/2.0))
        # 'remap' the path
        self.path = transform.map(arrowPath)
        self.setPath(self.path)
        # set position of the arrow
        self.setPos(x, y)