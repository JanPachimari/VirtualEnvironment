# basic imports
import os
import numpy as np
import pyqtgraph as qg
import PyQt5 as qt
# change working dictionary
#os.chdir("")
# import framework modules
from frontends.frontends_blender import FrontendBlenderInterface
from observations.image_observations import ImageObservationBaseline

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'!
visualOutput = True

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
    modules['world']=FrontendBlenderInterface('simple_grid_graph_env/simple_grid_graph_maze.blend', 'D:/blender_2.79b/blender')
    modules['observation']=ImageObservationBaseline(modules['world'],mainWindow,visualOutput)
    # move agent and change light color
    R2G = np.array([np.arange(1, 0, -0.01), np.arange(0, 1, 0.01), np.arange(1, 0, -0.01)*0])
    G2B = np.array([np.arange(1, 0, -0.01)*0, np.arange(1, 0, -0.01), np.arange(0, 1, 0.01)])
    B2R = np.array([np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)*0, np.arange(1, 0, -0.01)])
    for i in range(5):
        for j in range(R2G.shape[1]):
            modules['world'].setIllumination('Sun', R2G[:,j])
            modules['world'].step_simulation_without_physics(j/100, 0.0, 0.0)
            modules['observation'].update()
            if qt.QtGui.QApplication.instance() is not None:
                qt.QtGui.QApplication.instance().processEvents()
        for j in range(G2B.shape[1]):
            modules['world'].setIllumination('Sun', G2B[:,j])
            modules['world'].step_simulation_without_physics(j/100, 0.0, 0.0)
            modules['observation'].update()
            if qt.QtGui.QApplication.instance() is not None:
                qt.QtGui.QApplication.instance().processEvents()
        for j in range(B2R.shape[1]):
            modules['world'].setIllumination('Sun', B2R[:,j])
            modules['world'].step_simulation_without_physics(j/100, 0.0, 0.0)
            modules['observation'].update()
            if qt.QtGui.QApplication.instance() is not None:
                qt.QtGui.QApplication.instance().processEvents()
    # close blender
    modules['world'].stopBlender()
    # close GUI
    if visualOutput:
        mainWindow.close()

if __name__ == "__main__":    
    singleRun()