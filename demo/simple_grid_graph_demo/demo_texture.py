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

    world.set_renderState('barrier000-001', 'True')
    world.set_renderState('barrier001-002', 'True')
    world.set_renderState('barrier002-003', 'True')
    world.set_renderState('barrier004-005', 'True')
    world.set_renderState('barrier005-006', 'True')
    world.set_renderState('barrier006-007', 'True')
    world.set_renderState('barrier008-009', 'True')
    world.set_renderState('barrier009-010', 'True')
    world.set_renderState('barrier010-011', 'True')
    world.set_renderState('barrier012-013', 'True')
    world.set_renderState('barrier013-014', 'True')
    world.set_renderState('barrier014-015', 'True')

    textures = ["//textures/wall_01.bmp", "//textures/wall_02.bmp", "//textures/wall_03.bmp", "//textures/wall_04.bmp"]

    world.set_texture('barrier002-003', textures[0])
    world.set_texture('barrier001-002', textures[0])
    world.set_texture('barrier000-001', textures[0])

    world.set_texture('barrier006-007', textures[1])
    world.set_texture('barrier005-006', textures[1])
    world.set_texture('barrier004-005', textures[1])

    world.set_texture('barrier010-011', textures[2])
    world.set_texture('barrier009-010', textures[2])
    world.set_texture('barrier008-009', textures[2])

    world.set_texture('barrier014-015', textures[3])
    world.set_texture('barrier013-014', textures[3])
    world.set_texture('barrier012-013', textures[3])

    modules['observation'].update()

    for i in range(10000):
        world.step_simulation_without_physics(i / 100, 0.0, 0.0)
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
