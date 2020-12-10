# coding: utf-8

import rospy
import grasp_only.vrep as vrep
import os,time, random
import numpy as np
from grasp_only.srv import AddObjects,AddObjectsResponse, AddOneObject, AddOneObjectResponse
from std_srvs.srv import Empty,EmptyResponse

# Define colors for object meshes (Tableau palette)
color_space = np.asarray([[78.0, 121.0, 167.0],  # blue
                               [89.0, 161.0, 79.0],  # green
                               [156, 117, 95],  # brown
                               [242, 142, 43],  # orange
                               [237.0, 201.0, 72.0],  # yellow
                               [186, 176, 172],  # gray
                               [255.0, 87.0, 89.0],  # red
                               [176, 122, 161],  # purple
                               [118, 183, 178],  # cyan
                               [255, 157, 167]]) / 255.0  # pink
ipVREP = '127.0.0.1'


class ObjectsVREP(object):
    def __init__(self,workspace_limits, objects_dir):
        s = rospy.Service('add_objects', AddObjects, self.add_objects)
        s = rospy.Service('add_one_cube', AddObjects, self.add_one_cube)
        s = rospy.Service('add_one_object', AddOneObject, self.add_one_object)
        s = rospy.Service('move_object', Empty, self.move_object)
        self.sim_client = vrep.simxStart(ipVREP, 20001, True, True, 5000, 5)  # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation on port {}'.format(self.sim_client))
        self.workspace_limits = workspace_limits
        # Read files in object mesh directory
        self.obj_mesh_dir = objects_dir+'/blocks'
        self.new_objects_dir = objects_dir+'/newblocks'
        self.object_handles = []

    def get_obj_positions(self):
        """
        Retrive the list of all the objects in the workspace
        :return:
        """
        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
        return obj_positions

    def add_object(self, object_position, object_orientation, object_color, curr_mesh_file):
        """
        Used by add_objects and add_one_cube methods to add an object to VREP and update object_handles list.
        :param object_position:
        :param object_orientation:
        :param object_color:
        :param curr_mesh_file:
        :return:
        """
        curr_shape_name = 'shape_%02d' % len(self.object_handles)
        ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShape',
                                                                                          [0,0,255,0], object_position + object_orientation + object_color,
                                                                                          [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
        if ret_resp == 8:
            print('Failed to add new objects to simulation. Please restart.')
            exit()
        curr_shape_handle = ret_ints[0]
        self.object_handles.append(curr_shape_handle)
        time.sleep(2)

    # Gestion des services

    def move_object(self, req):
        """
        Used to remove an object from the gripper and put it outside the workspace.
        The one which is removed is the one with the highest elevation.
        :return:
        """
        object_positions = np.asarray(self.get_obj_positions())
        object_positions = object_positions[:, 2]
        grasped_object_ind = np.argmax(object_positions)
        grasped_object_handle = self.object_handles[grasped_object_ind]
        vrep.simxSetObjectPosition(self.sim_client, grasped_object_handle, -1,
                                   (-0.5, 0.5 + 0.05 * float(grasped_object_ind), 0.1), vrep.simx_opmode_blocking)
        return EmptyResponse()

    def add_objects(self,req):
        num_obj = req.nb_obj
        # Randomly choose objects to add to scene
        mesh_list = os.listdir(os.path.abspath(self.obj_mesh_dir))
        obj_mesh_ind = np.random.randint(0, len(mesh_list), size=num_obj)
        obj_mesh_color = color_space[np.asarray(range(num_obj)) % 10, :]
        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        for object_idx in range(num_obj):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, mesh_list[obj_mesh_ind[object_idx]])
            curr_mesh_file = os.path.abspath(curr_mesh_file)
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            object_color = [obj_mesh_color[object_idx][0], obj_mesh_color[object_idx][1], obj_mesh_color[object_idx][2]]
            self.add_object(object_position, object_orientation, object_color, curr_mesh_file)
        return AddObjectsResponse()

    def add_one_object(self,req):
        """
        Add one object described by its shape (the name of a file in the newblocks directory), its position, orientation and color.
        :param req: a AddOneObject message
        :return:
        """
        curr_mesh_file = self.new_objects_dir+'/' + req.shape + '.obj'  # cube, triangle, rectangle, longcylinder
        self.add_object(req.position, req.orientation, req.color, curr_mesh_file)
        return AddOneObjectResponse()

    def add_one_cube(self,req):
        """
        Add only one cube in the middle of the workspace ([-0.5, 0, 0.01] coordinates) for debug purpose.
        :param req:
        :return:
        """
        curr_mesh_file = self.new_objects_dir+'/cube.obj'  # a cube
        curr_mesh_file = os.path.abspath(curr_mesh_file)
        object_position = [-0.5, 0, 0.01] # in the middle of the workspace
        object_orientation = [0, 0, 0]
        object_color = [255, 0, 0]  # red
        self.add_object(object_position, object_orientation, object_color, curr_mesh_file)
        return AddObjectsResponse()