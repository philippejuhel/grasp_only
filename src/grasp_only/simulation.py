#!/usr/bin/env python
# coding: utf-8

import time
import os
import threading
import numpy as np
import cv2
import rospy
import array
from grasp_only.trainer import Trainer
from grasp_only.logger import Logger
from grasp_only.utils import *
from grasp_only.srv import CoordAction, InfoCamera
from grasp_only.srv import ColorDepthImages
from grasp_only.srv import AddObjects
from grasp_only.threadActionRunner import ThreadActionRunner
from std_srvs.srv import Empty

#
# Paramètres à ajuster
#
change_threshold = 300  # Nb min de pixels qui doivent changer pour que l'on considère q'un push a été efficace
min_height = 0.02  # Tous les pixels au dessus de cet altitude sont considérés comme faisant partie d'un objet
empty_threshold = 300 # Nb min de pixels en dessous duquel on considère qu'il n'y a plus d'objet dans la scène


class Simulation:
    def __init__(self, args):
        # --------------- Setup options ---------------
        self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])  # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        self.heightmap_resolution = args.heightmap_resolution  # Meters per pixel of heightmap
        self.num_obj = args.num_obj  # Nb of objects
        # ------------- Algorithm options -------------
        future_reward_discount = args.future_reward_discount
        explore_rate_decay = args.explore_rate_decay
        # ------ Pre-loading and logging options ------
        logging_directory = os.path.abspath(args.logging_directory)
        self.save_visualizations = args.save_visualizations  # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True
        # Set random seed
        np.random.seed(1234)
        # Setup virtual camera in simulation
        self.get_camera_informations()
        # Create objects and add them to VREP
        self.create_objects(self.num_obj)
        # Initialize trainer
        self.trainer = Trainer(future_reward_discount)
        # Initialize data logger
        self.logger = Logger(logging_directory)
        # logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
        self.logger.save_heightmap_info(self.workspace_limits, self.heightmap_resolution)  # Save heightmap parameters
        # Initialize variables for heuristic bootstrapping and exploration probability
        self.no_change_count = [2, 2]
        # Initialize exploration probability
        self.explore_prob = 0.5

    def execute(self):
        # Quick hack for nonlocal memory between threads in Python 2
        shared_variables = {'primitive_action': None,
                           'best_pix_ind': None,
                           'grasp_success': False}
        begin_training = False
        # -------------------------------------------------------------
        # Start main training/testing loop
        while True:
            print('\n%s iteration: %d' % ('Training', self.trainer.iteration))
            iteration_time_0 = time.time()
            # Get latest RGB-D image
            color_img, depth_img = self.get_camera_data()
            depth_img = depth_img * self.cam_depth_scale  # Apply depth scale from calibration
            # cv2.imshow('photo',depth_img)
            # if cv2.waitKey(0)==27:
            #     continue
            # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
            self.color_heightmap, self.depth_heightmap = get_heightmap(color_img, depth_img, self.cam_intrinsics, self.cam_pose, self.workspace_limits, self.heightmap_resolution)
            self.valid_depth_heightmap = self.depth_heightmap.copy()
            self.valid_depth_heightmap[np.isnan(self.valid_depth_heightmap)] = 0
            # Save RGB-D images and RGB-D heightmaps
            self.logger.save_images(self.trainer.iteration, color_img, depth_img, '0')
            self.logger.save_heightmaps(self.trainer.iteration, self.color_heightmap, self.valid_depth_heightmap, '0')
            # Reset simulation or pause real-world training if table is empty
            stuff_count = np.zeros(self.valid_depth_heightmap.shape)
            stuff_count[self.valid_depth_heightmap > min_height] = 1
            if np.sum(stuff_count) < empty_threshold or (self.no_change_count[0] + self.no_change_count[1] > 10): # Plus assez d'objets ou cela fait trop de push et de grasp sans succès
                self.no_change_count = [0, 0]  # ràz
                print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
                self.restart_sim()
                self.create_objects(self.num_obj)
                self.trainer.clearance_log.append([self.trainer.iteration])
                self.logger.write_to_log('clearance', self.trainer.clearance_log)
                continue
            # Run forward pass with network to get affordances
            grasp_predictions, state_feat = self.trainer.forward(self.color_heightmap, self.valid_depth_heightmap, is_volatile=True)
            # Execute best primitive action on robot in another thread
            thread_action_runner = ThreadActionRunner(self.explore_prob, self.trainer, self.logger,  grasp_predictions, self.workspace_limits, self.save_visualizations,
                                                      self.heightmap_resolution, self.valid_depth_heightmap, self.color_heightmap, shared_variables)
            thread_action_runner.start()
            # Run training iteration in current thread (aka training thread)
            if begin_training: # Pour ne pas faire le training dès la première image (on attendent d'en avoir au moins 2)
                self.train()
            # Sync both action thread and training thread
            thread_action_runner.join()
            # Save information for next training step
            begin_training = True # On a au moins pris une image, à la prochaine itération, on pourra lancer le training
            self.prev_color_heightmap = self.color_heightmap.copy()
            self.prev_depth_heightmap = self.depth_heightmap.copy()
            self.prev_valid_depth_heightmap = self.valid_depth_heightmap.copy()
            self.prev_grasp_predictions = grasp_predictions.copy()
            self.prev_grasp_success = shared_variables['grasp_success']
            self.prev_primitive_action = shared_variables['primitive_action']
            self.prev_best_pix_ind = shared_variables['best_pix_ind']
            self.trainer.iteration += 1
            iteration_time_1 = time.time()
            print('Time elapsed: %f' % (iteration_time_1 - iteration_time_0))

    def train(self):
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # Detect changes
        depth_diff = abs(self.depth_heightmap - self.prev_depth_heightmap)
        depth_diff[np.isnan(depth_diff)] = 0
        depth_diff[depth_diff > 0.3] = 0
        depth_diff[depth_diff < 0.01] = 0
        depth_diff[depth_diff > 0] = 1
        change_value = np.sum(depth_diff)  # How many pixels in [0.01cm , 0.3cm]
        change_detected = change_value > change_threshold or self.prev_grasp_success
        print('Change detected: %r (value: %d)' % (change_detected, change_value))
        if change_detected:
            self.no_change_count[1] = 0
        else:
            self.no_change_count[1] += 1
        # Compute training labels
        label_value, prev_reward_value = self.trainer.get_label_value(self.prev_primitive_action, self.prev_grasp_success, change_detected, self.color_heightmap, self.valid_depth_heightmap)
        self.trainer.label_value_log.append([label_value])
        self.logger.write_to_log('label-value', self.trainer.label_value_log)
        self.trainer.reward_value_log.append([prev_reward_value])
        self.logger.write_to_log('reward-value', self.trainer.reward_value_log)
        # Backpropagate
        self.trainer.backprop(self.prev_color_heightmap, self.prev_valid_depth_heightmap, self.prev_primitive_action, self.prev_best_pix_ind, label_value)
        # Adjust exploration probability
        self.explore_prob = max(0.5 * np.power(0.9998, self.trainer.iteration), 0.1)
        # Do sampling for experience replay
        sample_primitive_action = self.prev_primitive_action
        sample_primitive_action_id = 1
        sample_reward_value = 0 if prev_reward_value == 1 else 1
        # Get samples of the same primitive but with different results (commentaire pas exact? car results identiques)
        sample_ind = np.argwhere(np.logical_and(np.asarray(self.trainer.reward_value_log)[1:self.trainer.iteration, 0] == sample_reward_value,
                                                np.asarray(self.trainer.executed_action_log)[1:self.trainer.iteration, 0] == sample_primitive_action_id))
        if sample_ind.size > 0:
            # Find sample with highest surprise value
            sample_surprise_values = np.abs(np.asarray(self.trainer.predicted_value_log)[sample_ind[:, 0]] - np.asarray(self.trainer.label_value_log)[sample_ind[:, 0]])
            sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
            sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
            pow_law_exp = 2
            rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1) * (sample_ind.size - 1)))
            sample_iteration = sorted_sample_ind[rand_sample_ind]
            print('Experience replay: iteration %d (surprise value: %f)' % (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))
            # Load sample RGB-D heightmap
            self.sample_color_heightmap = cv2.imread(os.path.join(self.logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
            self.sample_color_heightmap = cv2.cvtColor(self.sample_color_heightmap, cv2.COLOR_BGR2RGB)
            sample_depth_heightmap = cv2.imread(os.path.join(self.logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
            sample_depth_heightmap = sample_depth_heightmap.astype(np.float32) / 100000
            # Compute forward pass with sample
            sample_grasp_predictions, sample_state_feat = self.trainer.forward(self.sample_color_heightmap, sample_depth_heightmap, is_volatile=True)
            # Get labels for sample and backpropagate
            sample_best_pix_ind = (np.asarray(self.trainer.executed_action_log)[sample_iteration, 1:4]).astype(int)
            self.trainer.backprop(self.sample_color_heightmap, sample_depth_heightmap, sample_primitive_action, sample_best_pix_ind, sample_reward_value)
            # Recompute prediction value
            self.trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]
        else:
            print('Not enough prior training samples. Skipping experience replay.')
        # Save model snapshot
        self.logger.save_backup_model(self.trainer.model)
        if self.trainer.iteration % 50 == 0:
            self.logger.save_model(self.trainer.iteration, self.trainer.model)
            if self.trainer.use_cuda:
                self.trainer.model = self.trainer.model.cuda()
        
        
    # Méthodes faisant le lien avec les autres objets en appelant leurs services

    def restart_sim(self):
        '''
        Appel au service restart_sim du node robot
        :return:
        '''
        rospy.wait_for_service('restart_sim')
        try:
            cmdRestartSim = rospy.ServiceProxy('restart_sim', Empty)
            resp = cmdRestartSim()
            if resp:
                print('Fin de restart_sim')
        except rospy.ServiceException as e:
            print("Service restart_sim call failed: %s" % e)

    def get_camera_data(self):
        rospy.wait_for_service('get_color_depth_images')
        try:
            getImages = rospy.ServiceProxy('get_color_depth_images', ColorDepthImages)
            resp = getImages()
            width = resp.width
            height = resp.height
            resp.colorImage = list(array.array("B", resp.colorImage))
            color_image = np.asarray(resp.colorImage, dtype=np.uint8).reshape(width, height, 3)
            depth_image = np.asarray(resp.depthImage, dtype=np.float64).reshape(width, height)
            return color_image, depth_image
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def get_camera_informations(self):
        rospy.wait_for_service('get_camera_informations')
        try:
            getInfos = rospy.ServiceProxy('get_camera_informations', InfoCamera)
            resp = getInfos()
            self.cam_depth_scale = resp.depth_scale
            self.cam_intrinsics = np.asarray(resp.intrinsics, dtype=np.float64).reshape(3, 3)
            self.cam_pose = np.asarray(resp.pose, dtype=np.float64).reshape(4, 4)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def create_objects(self,nb_obj):
        rospy.wait_for_service('add_objects')
        try:
            addObjects = rospy.ServiceProxy('add_objects', AddObjects)
            addObjects(nb_obj)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
