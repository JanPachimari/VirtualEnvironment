from observations.image_observations import ImageObservationBaseline
from frontends.frontends_blender import FrontendBlenderInterface
import os
import numpy as np
import pyqtgraph as qg
import PyQt5 as qt
# change working directory
os.chdir("C:/Users/Jan/Desktop/Study_project")
# import framework modules

visualOutput = True


def singleRun():
    '''
    This method performs a single experimental run to test methods implemented in this study project.
    '''

    np.random.seed()
    mainWindow = None
    if visualOutput:
        mainWindow = qg.GraphicsWindow(title="WorkingTitle_Framework")
    # A dictionary that contains all employed modules
    modules = dict()
    modules['world'] = FrontendBlenderInterface(
        'simple_grid_graph_env/simple_grid_graph_maze.blend',
        'D:/Blender Foundation/Blender/blender.exe')
    modules['observation'] = ImageObservationBaseline(
        modules['world'], mainWindow, visualOutput)

    # Testing custom functions
    world = modules['world']
    barriers = world.get_barrierIDs()
    for i in barriers:
        world.set_renderState(i, 'False')
        print(world.get_barrierInfo(i))

    world.set_spotlight('spotlight000', 'True')
    world.set_spotlight('spotlight003', 'True')
    world.set_spotlight('spotlight008', 'True')
    world.set_spotlight('spotlight011', 'True')
    world.set_spotlight('spotlight012', 'True')
    world.set_spotlight('spotlight013', 'True')
    world.set_spotlight('spotlight014', 'True')
    world.set_spotlight('spotlight015', 'True')

    modules['observation'].update()

    for i in range(10000):
        modules['world'].step_simulation_without_physics(i / 100, 0.0, 0.0)
        modules['observation'].update()
    if qt.QtGui.QApplication.instance() is not None:
        qt.QtGui.QApplication.instance().processEvents()

    input("Press enter to continue")

    # Close Blender
    modules['world'].stopBlender()
    # Close GUI
    if visualOutput:
        mainWindow.close()


if __name__ == "__main__":
    singleRun()
