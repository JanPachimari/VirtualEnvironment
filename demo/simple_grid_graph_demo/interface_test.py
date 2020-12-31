from observations.image_observations import ImageObservationBaseline
from frontends.frontends_blender import FrontendBlenderInterface
import os
import numpy as np
import pyqtgraph as qg
import PyQt5 as qt
# change working directory
os.chdir("C:/Users/yoric/Desktop/virtual-environment-interface")
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
        'simple_grid_graph_env/simple_grid_graph_maze.blend', 'C:/Users/yoric/blender-2.79b-windows64/blender')
    modules['observation'] = ImageObservationBaseline(
        modules['world'], mainWindow, visualOutput)

    # Testing custom functions
    barriers = modules['world'].get_barrierIDs()
    textures = ["//textures/wall_01.bmp",
                "//textures/wall_02.bmp", "//textures/wall_03.bmp", "//textures/wall_04.bmp"]
    # Setting render state, rotation and texture of barriers and printing barrier infos before and after
    j = 0
    for i in barriers:
        print(modules['world'].get_barrierInfo(i))
        modules['world'].set_renderState(i, bool(j % 2))
        modules['world'].set_rotation(i, ((90*j) % 360))
        modules['world'].set_texture(i, textures[j % 4])
        print(modules['world'].get_barrierInfo(i))
        j += 1
    # Rendering half the spotlights
    for i in range(16):
        spotlight = 'spotlight{:0>3}'.format('%d' % i)
        modules['world'].set_spotlight(spotlight, bool(i % 2))
    # Testing handling of invalid paths for set_texture
    modules['world'].set_texture(barriers[12], 'invalid_path')

    modules['observation'].update()

    for i in range(100):
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
