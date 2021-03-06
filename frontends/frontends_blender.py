# basic imports
import sys
import socket
import os
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R
# shapely
from shapely.geometry import Polygon


class FrontendBlenderInterface():
    '''
    The Blender interface class.
    This class connects to the Blender environment and controls the flow of commands/data that goes to/comes from the Blender environment.

    | **Args**
    | scenarioName:                 The path to the blender scene that should be loaded.
    | blenderExecutable:            The path to the blender executable (optional, if a path variable was defined).
    '''

    def __init__(self, scenarioName, blenderExecutable=None):
        # determine path of the blender executable
        self.BLENDER_EXECUTABLE = ''
        # check parameter
        if blenderExecutable is None:
            path = os.environ['BLENDER_EXECUTABLE_PATH']
            self.BLENDER_EXECUTABLE = path + 'blender'
        else:
            self.BLENDER_EXECUTABLE = blenderExecutable
        # determine current path
        paths = os.environ['PYTHONPATH'].split(':')
        path = ''
        for p in paths:
            if 'CoBeL-RL' in p:
                path = p
        # determine path to the blender scene
        scenarioName = path + '/environments/environments_blender/' + scenarioName
        # start blender subprocess
        subprocess.Popen([self.BLENDER_EXECUTABLE, scenarioName, '--window-border',
                          '--window-geometry', '1320', '480', '600', '600', '--enable-autoexec'])
        # prepare sockets for communication
        self.controlSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.videoSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dataSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # wait for BlenderControl to start, this socket take care of command/data(limited amount) transmission
        BlenderControlAvailable = False
        while not BlenderControlAvailable:
            try:
                self.controlSocket.connect(('localhost', 5000))
                self.controlSocket.setblocking(1)
                BlenderControlAvailable = True
            except:
                pass
        print('Blender control connection has been initiated.')
        self.controlSocket.setsockopt(
            socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # wait for BlenderVideo to start, this socket transmits video data from Blender
        BlenderVideoAvailable = False
        while not BlenderVideoAvailable:
            try:
                self.videoSocket.connect(('localhost', 5001))
                self.videoSocket.setblocking(1)
                BlenderVideoAvailable = True
            except:
                pass
        print('Blender video connection has been initiated.')
        self.videoSocket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # wait for BlenderData to start, this socket is currently legacy, can probably be removed in future system versions(?)
        BlenderDataAvailable = False
        while not BlenderDataAvailable:
            try:
                self.dataSocket.connect(('localhost', 5002))
                self.dataSocket.setblocking(1)
                BlenderDataAvailable = True
            except:
                pass

        print('Blender data connection has been initiated.')
        self.dataSocket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # sockets are now open!

        # get the maximum safe zone dimensions for the robot to move in
        self.controlSocket.send('getSafeZoneDimensions'.encode('utf-8'))
        value = self.controlSocket.recv(1000).decode('utf-8')
        [minXStr, minYStr, minZStr, maxXStr,
            maxYStr, maxZStr] = value.split(',')
        # store the environment limits
        self.minX = float(minXStr)
        self.minY = float(minYStr)
        self.minZ = float(minZStr)
        self.maxX = float(maxXStr)
        self.maxY = float(maxYStr)
        self.maxZ = float(maxZStr)
        print('Safe Zone dimensions retrieved.')

        # get the safe zone layout for the robot to move in
        self.controlSocket.send('getSafeZoneLayout'.encode('utf-8'))
        value = self.controlSocket.recv(1000).decode('utf-8')
        self.controlSocket.send('AKN'.encode('utf-8'))
        nrOfVertices = int(value)
        print('Safe Zone layout retrieved.')

        # temporary variables for extraction of the environment's perimeter
        vertexList, safeZoneSegmentList = [], []

        for i in range(nrOfVertices):
            value = self.controlSocket.recv(1000).decode('utf-8')
            xStr, yStr = value.split(',')
            x = float(xStr)
            y = float(yStr)
            self.controlSocket.send('AKN'.encode('utf-8'))
            # update the vertex list
            vertexList = vertexList + [[x, y]]
            j = i + 1
            if i == nrOfVertices - 1:
                j = 0
            # update the segment list
            safeZoneSegmentList += [[i, j]]

        # convert the above lists to numpy arrays
        self.safeZoneVertices = np.array(vertexList)
        self.safeZoneSegments = np.array(safeZoneSegmentList)

        # construct the safe polygon from the above lists
        self.safeZonePolygon = Polygon(vertexList)
        self.controlSocket.recv(100).decode('utf-8')

        # initial robot pose
        self.robotPose = np.array([0.0, 0.0, 1.0, 0.0])
        # stores the actual goal position (coordinates of the goal node)
        self.goalPosition = None
        # indicator that flags arrival of the agent/robot at the actual goal node (this need NOT be the global goal node!)
        self.goalReached = False

        # a dict that stores environmental information in each time step
        self.envData = dict()
        self.envData['timeData'] = None
        self.envData['poseData'] = None
        self.envData['sensorData'] = None
        self.envData['imageData'] = None

        print('Frontend set up.')

        # propel the simulation for 10 timesteps to finish initialization of the simulation framework
        for i in range(10):
            self.step_simulation_without_physics(0.0, 0.0, 0.0)
        print('Framework initialized.')

    def getManuallyDefinedTopologyNodes(self):
        '''
        This function reads all manually defined topology nodes from the environment (if such nodes are defined).
        '''
        # get the forbidden zones' layouts for the robot to avoid
        self.controlSocket.send(
            'getManuallyDefinedTopologyNodes'.encode('utf-8'))
        nrOfNodesStr = self.controlSocket.recv(1000).decode('utf-8')
        nrOfNodes = int(nrOfNodesStr)
        print('Received %d manually defined topology nodes' % nrOfNodes)
        self.controlSocket.send('AKN'.encode('utf-8'))
        manuallyDefinedTopologyNodes = []
        # retrieve individual noes
        for n in range(nrOfNodes):
            nodeStr = self.controlSocket.recv(1000).decode('utf-8')
            # print('reading manually defined node: %s' % nodeStr)
            nameStr, xStr, yStr, typeStr = nodeStr.split(',')
            nodeName = nameStr
            nodeX = float(xStr)
            nodeY = float(yStr)
            nodeType = typeStr
            self.controlSocket.send('AKN'.encode('utf-8'))
            manuallyDefinedTopologyNodes += [[nodeName,
                                              nodeX, nodeY, nodeType]]
        # wait for AKN
        self.controlSocket.recv(1000).decode('utf-8')

        return manuallyDefinedTopologyNodes

    def getManuallyDefinedTopologyEdges(self):
        '''
        This function reads all manually defined topology edges from the environment (if such edges are defined).
        '''
        # get the forbidden zones' layouts for the robot to avoid
        self.controlSocket.send(
            'getManuallyDefinedTopologyEdges'.encode('utf-8'))
        nrOfEdgesStr = self.controlSocket.recv(1000).decode('utf-8')
        nrOfEdges = int(nrOfEdgesStr)
        print('Received %d manually defined topology edges' % nrOfEdges)
        self.controlSocket.send('AKN'.encode('utf-8'))
        manuallyDefinedTopologyEdges = []
        # retrieve individual edges
        for n in range(nrOfEdges):
            edgeStr = self.controlSocket.recv(1000).decode('utf-8')
            # print('reading manually defined edge: %s' % edgeStr)
            nameStr, firstStr, secondStr = edgeStr.split(',')
            edgeName = nameStr
            first = int(firstStr)
            second = int(secondStr)
            self.controlSocket.send('AKN'.encode('utf-8'))
            manuallyDefinedTopologyEdges += [[edgeName, first, second]]
        # wait for AKN
        self.controlSocket.recv(1000).decode('utf-8')

        return manuallyDefinedTopologyEdges

    def recvImage(self, s, imgSize):
        '''
        This function reads an image chunk from a socket.

        | **Args**
        | s:                            The socket to read image from.
        | imgSize:                      The size of the image to read.
        '''
        # define buffer
        imgData = b''
        # define byte 'counter'
        receivedBytes = 0
        # read the image in chunks
        while receivedBytes < imgSize:
            dataChunk = s.recv(imgSize - receivedBytes)
            if dataChunk == '':
                break
            receivedBytes += len(dataChunk)
            imgData += dataChunk

        return imgData

    def stepSimulation(self, velocityLinear, omega):
        '''
        This function propels the simulation. It uses physics to guide the agent/robot with linear and angular velocities.

        | **Args**
        | velocityLinear:               The requested linear (translational) velocity of the agent/robot.
        | omega:                        The requested rotational (angular) velocity of the agent/robot.
        '''
        # the basic capture image size for the images taken by the robot's omnicam, the width of the omnicam image is actually 4*capAreaWidth
        capAreaWidth, capAreaHeight = 64, 64
        # from the linear/angular velocities, compute the left and right wheel velocities
        [velocityLeft, velocityRight] = self.setRobotVelocities(
            velocityLinear, omega)
        # send the actuation command to the virtual robot/agent
        sendStr = 'stepSimulation,%f,%f' % (velocityLeft, velocityRight)
        self.controlSocket.send(sendStr.encode('utf-8'))
        # retrieve images from all cameras of the robot (front, left, right, back)
        # front
        imgFront = self.recvImage(
            self.videoSocket, capAreaWidth*capAreaHeight*4)
        imgNPFront = np.fromstring(imgFront, dtype=np.uint8)
        imgNPFront = imgNPFront.reshape((capAreaHeight, capAreaWidth, 4))
        # left
        imgLeft = self.recvImage(
            self.videoSocket, capAreaWidth*capAreaHeight*4)
        imgNPLeft = np.fromstring(imgLeft, dtype=np.uint8)
        imgNPLeft = imgNPLeft.reshape((capAreaHeight, capAreaWidth, 4))
        # right
        imgRight = self.recvImage(
            self.videoSocket, capAreaWidth*capAreaHeight*4)
        imgNPRight = np.fromstring(imgRight, dtype=np.uint8)
        imgNPRight = imgNPRight.reshape((capAreaHeight, capAreaWidth, 4))
        # back
        imgBack = self.recvImage(
            self.videoSocket, capAreaWidth*capAreaHeight*4)
        imgNPBack = np.fromstring(imgBack, dtype=np.uint8)
        imgNPBack = imgNPBack.reshape((capAreaHeight, capAreaWidth, 4))
        # and construct the omnicam image from the single images.
        # Note: the images are just 'stitched' together, no distortion correction takes place (so far).
        imageData = np.hstack((imgNPFront, imgNPRight, imgNPBack, imgNPLeft))
        # center(?) the omnicam image
        imageData = np.roll(imageData, 96, axis=1)
        # extract RGB information channels
        imageData = imageData[:, :, :3]
        # retrieve telemetry data from the virtual agent/robot
        telemetryDataString = self.controlSocket.recv(2000).decode('utf-8')
        [timeDataStr, poseDataStr,
            sensorDataStr] = telemetryDataString.split(':')
        # extract time data from telemetry
        timeData = float(timeDataStr)
        # extract pose data from telemetry
        [x, y, nx, ny] = poseDataStr.split(',')
        poseData = [float(x), float(y), float(nx), float(ny)]
        # extract sensor data from telemetry
        [s0, s1, s2, s3, s4, s5, s6, s7] = sensorDataStr.split(',')
        sensorData = np.array([float(s0), float(s1), float(s2), float(
            s3), float(s4), float(s5), float(s6), float(s7)], dtype='float')
        # update robot's/agent's pose
        self.robotPose = np.array(poseData)
        # update environmental information
        self.envData['timeData'] = timeData
        self.envData['poseData'] = poseData
        self.envData['sensorData'] = sensorData
        self.envData['imageData'] = imageData

        return [timeData, poseData, sensorData, imageData]

    def step_simulation_without_physics(self, newX, newY, newYaw):
        '''
        This function propels the simulation. It uses teleportation to guide the agent/robot directly by means of global x,y,yaw values.

        | **Args**
        | newX:                         The global x position to teleport to.
        | newY:                         The global y position to teleport to.
        | newYaw:                       The global yaw value to rotate to.
        '''
        # the basic capture image size for the images taken by the robot's omnicam, the width of the omnicam image is actually 4*capAreaWidth
        capAreaWidth, capAreaHeight = 64, 64
        # send the actuation command to the virtual robot/agent
        sendStr = 'stepSimNoPhysics,%f,%f,%f' % (newX, newY, newYaw)
        self.controlSocket.send(sendStr.encode('utf-8'))
        # retrieve images from all cameras of the robot (front, left, right, back)
        # front
        imgFront = self.recvImage(
            self.videoSocket, capAreaWidth*capAreaHeight*4)
        imgNPFront = np.fromstring(imgFront, dtype=np.uint8)
        imgNPFront = imgNPFront.reshape((capAreaHeight, capAreaWidth, 4))
        # left
        imgLeft = self.recvImage(
            self.videoSocket, capAreaWidth*capAreaHeight*4)
        imgNPLeft = np.fromstring(imgLeft, dtype=np.uint8)
        imgNPLeft = imgNPLeft.reshape((capAreaHeight, capAreaWidth, 4))
        # right
        imgRight = self.recvImage(
            self.videoSocket, capAreaWidth*capAreaHeight*4)
        imgNPRight = np.fromstring(imgRight, dtype=np.uint8)
        imgNPRight = imgNPRight.reshape((capAreaHeight, capAreaWidth, 4))
        # back
        imgBack = self.recvImage(
            self.videoSocket, capAreaWidth*capAreaHeight*4)
        imgNPBack = np.fromstring(imgBack, dtype=np.uint8)
        imgNPBack = imgNPBack.reshape((capAreaHeight, capAreaWidth, 4))
        # and construct the omnicam image from the single images.
        # Note: the images are just 'stitched' together, no distortion correction takes place (so far).
        imageData = np.hstack((imgNPFront, imgNPRight, imgNPBack, imgNPLeft))
        # center(?) the omnicam image
        imageData = np.roll(imageData, 96, axis=1)
        # extract RGB information channels
        imageData = imageData[:, :, :3]
        # retrieve telemetry data from the virtual agent/robot
        telemetryDataString = self.controlSocket.recv(2000).decode('utf-8')
        [timeDataStr, poseDataStr,
            sensorDataStr] = telemetryDataString.split(':')
        # extract time data from telemetry
        timeData = float(timeDataStr)
        # extract pose data from telemetry
        [x, y, nx, ny] = poseDataStr.split(',')
        poseData = [float(x), float(y), float(nx), float(ny)]
        # extract sensor data from telemetry
        [s0, s1, s2, s3, s4, s5, s6, s7] = sensorDataStr.split(',')
        sensorData = np.array([float(s0), float(s1), float(s2), float(
            s3), float(s4), float(s5), float(s6), float(s7)], dtype='float')
        # update robot's/agent's pose
        self.robotPose = np.array(poseData)
        # update environmental information
        self.envData['timeData'] = timeData
        self.envData['poseData'] = poseData
        self.envData['sensorData'] = sensorData
        self.envData['imageData'] = imageData

        return [timeData, poseData, sensorData, imageData]

    def actuateRobot(self, actuatorCommand):
        '''
        This function actually actuates the agent/robot in the virtual environment.

        | **Args**
        | actuatorCommand:              The command that is used in the actuation process.
        '''
        # if the actuator command has more than 2 array entries, this is a teleport command,
        # and will cause a teleport jump of the agent/robot (no physics support)
        if actuatorCommand.shape[0] > 2:
            # call the teleportation routine
            [timeData, poseData, sensorData, imageData] = self.step_simulation_without_physics(
                actuatorCommand[0], actuatorCommand[1], actuatorCommand[2])
            # flag if the robot reached the goal (should always be the case after a teleportation)
            if self.goalPosition is not None:
                print(np.linalg.norm(poseData[:2] - self.goalPosition))
                if np.linalg.norm(poseData[:2] - self.goalPosition) < 0.01:
                    self.goalReached = True
            # return the data acquired from the robot/agent/environment
            return timeData, poseData, sensorData, imageData
        else:
            # otherwise, this is a standard motion command with physics support (the agent/robot approaches the goal by actuating the robot's/agent's wheels)
            [timeData, poseData, sensorData, imageData] = self.stepSimulation(
                actuatorCommand[0], actuatorCommand[1])
            # flag if the robot/agent reached the goal already
            if self.goalPosition is not None:
                if np.linalg.norm(poseData[:2] - self.goalPosition) < 0.01:
                    self.goalReached = True
            # return the data acquired from the robot/agent/environment
            return timeData, poseData, sensorData, imageData

    def setXYYaw(self, pose):
        '''
        This function teleports the robot to a novel pose.

        | **Args**
        | pose:                         The pose to teleport to.
        '''
        # send the request for teleportation to Blender
        poseStr = '%f,%f,%f' % (pose[0], pose[1], pose[2])
        sendStr = 'setXYYaw,robotSupport,%s' % poseStr
        self.controlSocket.send(sendStr.encode('utf-8'))
        # wait for acknowledge from Blender
        self.controlSocket.recv(50)

    def setIllumination(self, lightSource, color):
        '''
        This function sets the color of a dedicated light source.

        | **Args**
        | lightSource:                  The name of the light source to change.
        | color:                        The color of the light source (as [red,green,blue]).
        '''
        # send the request for illumination change to Blender
        illuminationStr = '%s,%f,%f,%f' % (
            lightSource, color[0], color[1], color[2])
        sendStr = 'setIllumination,%s' % illuminationStr
        self.controlSocket.send(sendStr.encode('utf-8'))
        # wait for acknowledge from Blender
        self.controlSocket.recv(50)

    def getLimits(self):
        '''
        This function returns the limits of the environmental perimeter.
        '''
        return np.array([[self.minX, self.maxX], [self.minY, self.maxY]])

    def getWallGraph(self):
        '''
        This function returns the environmental perimeter by means of wall vertices/segments.
        '''
        return self.safeZoneVertices, self.safeZoneSegments

    def stopBlender(self):
        '''
        This function shuts down Blender.
        '''
        try:
            self.controlSocket.send('stopSimulation'.encode('utf-8'))
        except:
            print(sys.exc_info()[1])

            #####################################################################################
            ##                                                                                 ##
            ##     Anything beyond this point has been added as part of the study project      ##
            ## 'Development of an interface to control the structure of a virtual environment' ##
            ##                                                                                 ##
            #####################################################################################

    def set_renderState(self, barrier_id, renderState):
        '''
        This function sets the render state of a given barrier on the topology graph to true/false.
        The main loop will also take care of adjusting the topology graph accordingly.
        | **Args**
        | barrier_id:   The id of the barrier to be set
        | renderState:  True/False
        '''

        contentStr = '%s,%r' % (barrier_id, renderState)
        sendStr = 'set_renderState,%s' % contentStr
        self.controlSocket.send(sendStr.encode('utf-8'))
        # Waiting for acknowledgement from Blender
        self.controlSocket.recv(50)

    def set_rotation(self, barrier_id, rotation):
        '''
        This function sets the rotation of a given barrier
        | **Args**
        | barrier_id:   The id of the barrier to be rotated
        | rotation:     The rotation in degrees (0-360)
        '''

        contentStr = '%s,%f' % (barrier_id, rotation)
        sendStr = 'set_rotation,%s' % contentStr
        self.controlSocket.send(sendStr.encode('utf-8'))
        # Waiting for acknowledgement from Blender
        self.controlSocket.recv(50)

    def set_texture(self, barrier_id, texture):
        '''
        This function sets the texture of the barrier
        | **Args**
        | barrier_id:   The id of the barrier to be given a texture
        | texture:      Filepath to the chosen texture, relative to the folder containing the environment file (.blend).
        '''

        contentStr = '%s,%s' % (barrier_id, texture)
        sendStr = 'set_texture,%s' % contentStr
        self.controlSocket.send(sendStr.encode('utf-8'))
        # Waiting for acknowledgement from Blender
        self.controlSocket.recv(50)

    def set_barrier(self, barrier_id, render_state, rotation, texture):
        '''
        This function calls all the barrier defining functions for the given barrier
        | **Args**
        | barrier_id:   The id of the barrier to be set
        | render_state: Boolean for setting wether the given barrier should be rendered
        | rotation:     The rotation in degrees
        | texture:      Filepath to the chosen texture, relative to the blender executable
        '''

        print('Setting barrier with barrierID', barrier_id, '.')
        self.set_renderState(barrier_id, render_state)
        self.set_rotation(barrier_id, rotation)
        self.set_texture(barrier_id, texture)

    def get_barrierInfo(self, barrier_id):
        '''
        This function returns a dictionary containing renderState, rotation and texture of a given barrier
        | **Args**
        | barrier_id:   The id of the barrier whose information is to be retrieved
        '''
        # Sending command
        sendStr = 'get_barrierInfo,%s' % barrier_id
        self.controlSocket.send(sendStr.encode('utf-8'))
        # Receiving data
        responseStr = self.controlSocket.recv(3000).decode('utf-8')
        responseList = responseStr.split(',')

        # Decoding string to rotation matrix
        rotationList = responseList[1]
        rotationArray = rotationList.split('|')
        rotationMatrix = [i.split(';') for i in rotationArray]
        # Converting rotation matrix to Euler angles
        r = R.from_matrix(rotationMatrix)
        rotation = r.as_euler('xyz', degrees=True)

        # Saving results in dictionary
        barrierInfo = {
            'renderState': responseList[0],
            'rotation': rotation[2],
            'texture': responseList[2]
        }
        return barrierInfo

    def get_barrierIDs(self):
        '''
        This function returns a list of all barrier objects.
        BarrierIDs are in the from "barrierxxx-yyy", where xxx and yyy are the
        numbers of the nodes the barrier is standing between.
        This may be subject to change.
        | **Args**
        | barrier_id:   The id of the barrier to be set
        '''

        sendStr = 'get_barrierIDs'
        self.controlSocket.send(sendStr.encode('utf-8'))
        barrierStr = self.controlSocket.recv(1000).decode('utf-8')

        barrierIDs = barrierStr.split(',')
        return barrierIDs

    def set_spotlight(self, spotlight_id, renderState):
        '''
        This function sets the render state of a given spotlight object.
        A spotlight lights up the area around a topology graph node.
        | **Args**
        | spotlight_id: The id of the spotlight to be toggled
        | renderState:  True/False
        '''

        contentStr = '%s,%r' % (spotlight_id, renderState)
        sendStr = 'set_spotlight,%s' % contentStr
        self.controlSocket.send(sendStr.encode('utf-8'))
        # Waiting for acknowledgement from Blender
        self.controlSocket.recv(50)
