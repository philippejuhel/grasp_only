# coding: utf-8

import rospy
import grasp_only.vrep as vrep
import math
import numpy as np
from grasp_only.srv import ColorDepthImages, ColorDepthImagesResponse, InfoCamera, InfoCameraResponse


class CameraVREP(object):
    def __init__(self):
        s = rospy.Service('get_color_depth_images', ColorDepthImages, self.get_data_for_service)
        s = rospy.Service('get_camera_informations', InfoCamera, self.get_informations_for_service)
        ipVREP = '127.0.0.1'
        self.sim_client = vrep.simxStart(ipVREP, 20002, True, True, 5000, 5)  # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation on port {}'.format(self.sim_client))
        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)
        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4, 4)
        cam_rotm[0:3, 0:3] = np.linalg.inv(self.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representation camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1.0
        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        print(self.bg_color_img.shape)
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

    def get_informations_for_service(self, req):
        """
        Called by the 'get_camera_informations' service to retrieve the webcam instrinsic parameters
        and its pose.
        :param req:
        :return:
        """
        intrinsics_1d = tuple(self.cam_intrinsics.reshape(1, -1)[0])
        pose_1d = tuple(self.cam_pose.reshape(1, -1)[0])
        return InfoCameraResponse(self.cam_depth_scale, intrinsics_1d, pose_1d)

    def get_data_for_service(self, req):
        """
        Called by the 'get_color_depth_images' service to retrieve the RGB and depth images.
        :param req:
        :return:
        """
        color_img, depth_img = self.get_camera_data()
        h, w, c = color_img.shape
        color_numpy_1d = tuple(color_img.reshape(1, -1)[0])
        depth_numpy_1d = tuple(depth_img.reshape(1, -1)[0])
        print('conversion OK')
        return ColorDepthImagesResponse(h, w, color_numpy_1d, depth_numpy_1d)

    def get_camera_data(self):
        """
        Get the RGB and depth images from the simulated camera.
        :return:
        """
        # Get color image from simulation
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float) / 255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)
        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)  #, dtype=np.float64)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        z_near = 0.01
        z_far = 10
        depth_img = depth_img * (z_far - z_near) + z_near
        return color_img, depth_img

    def euler2rotm(self, theta):
        """
        Get rotation matrix from euler angles
        :param theta:
        :return:
        """
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])
        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])
        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R
