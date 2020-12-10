# coding: utf-8

import time
import numpy as np
from threading import Thread, Event
import cv2
import rospy
from grasp_only.srv import CoordAction

class ThreadActionRunner(Thread):
    def __init__(self, explore_prob, trainer, logger, grasp_predictions, workspace_limits, save_visualizations, heightmap_resolution, valid_depth_heightmap, color_heightmap, output_variables):
        Thread.__init__(self)
        self.explore_prob = explore_prob
        self.grasp_predictions = grasp_predictions
        self.output_variables = output_variables
        self.trainer = trainer
        self.logger = logger
        self.workspace_limits = workspace_limits
        self.valid_depth_heightmap = valid_depth_heightmap
        self.color_heightmap = color_heightmap
        self.heightmap_resolution = heightmap_resolution
        self.save_visualizations = save_visualizations

    def run(self):
        # Determine whether grasping or pushing should be executed based on network predictions
        best_grasp_conf = np.max(self.grasp_predictions)
        print('Primitive confidence scores:  %f (grasp)' % (best_grasp_conf))
        explore_actions = np.random.uniform() < self.explore_prob
        if explore_actions:  # Exploration (do random action)
            print('Strategy: explore (exploration probability: %f)' % self.explore_prob)
        else: # Exploitation (do best action)
            print('Strategy: exploit (exploration probability: %f)' % self.explore_prob)
        action_to_execute = 'grasp'
        self.trainer.is_exploit_log.append([0 if explore_actions else 1])
        self.logger.write_to_log('is-exploit', self.trainer.is_exploit_log)
        # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x) => where should the action take place
        best_pix_ind = np.unravel_index(np.argmax(self.grasp_predictions), self.grasp_predictions.shape)
        predicted_value = np.max(self.grasp_predictions)
        print('Action: %s at (%d, %d, %d)' % (action_to_execute, best_pix_ind[0], best_pix_ind[1],best_pix_ind[2]))
        # Save predicted confidence value
        self.trainer.predicted_value_log.append([predicted_value])
        self.logger.write_to_log('predicted-value', self.trainer.predicted_value_log)
        # Compute 3D position of pixel
        best_rotation_angle = np.deg2rad(best_pix_ind[0] * (360.0 / self.trainer.model.num_rotations))
        best_pix_x = best_pix_ind[2]
        best_pix_y = best_pix_ind[1]
        primitive_position = [best_pix_x * self.heightmap_resolution + self.workspace_limits[0][0],
                              best_pix_y * self.heightmap_resolution + self.workspace_limits[1][0],
                              self.valid_depth_heightmap[best_pix_y][best_pix_x] + self.workspace_limits[2][0]]
        # Save executed primitive
        self.trainer.executed_action_log.append([1, best_pix_ind[0], best_pix_ind[1], best_pix_ind[2]])   #  1 - grasp
        self.logger.write_to_log('executed-action', self.trainer.executed_action_log)
        # Visualize executed primitive, and affordances
        if self.save_visualizations:
            grasp_pred_vis = self.trainer.get_prediction_vis(self.grasp_predictions, self.color_heightmap, best_pix_ind)
            self.logger.save_visualizations(self.trainer.iteration, grasp_pred_vis, 'grasp')
            cv2.imwrite('visualization.grasp.png', grasp_pred_vis)
        # Execute primitive
        self.output_variables['grasp_success'] = self.grasp(primitive_position, best_rotation_angle)
        print('Grasp successful: %r' % (self.output_variables['grasp_success']))
        # Output data for main thread (mainSimulation)
        self.output_variables['primitive_action'] = action_to_execute
        self.output_variables['best_pix_ind'] = best_pix_ind
        time.sleep(0.01)

    def grasp(self, position, angle):
        '''
        Appel au service robot_grasp du node robot
        :param position:
        :param angle:
        :return:
        '''
        rospy.wait_for_service('robot_grasp')
        try:
            cmdGrasp = rospy.ServiceProxy('robot_grasp', CoordAction)
            resp = cmdGrasp(position, angle)
            print('===============> robot_grasp : {},{} => {}'.format(position, angle, resp))
            return resp.success
        except rospy.ServiceException as e:
            print("Service robot_grasp call failed: %s" % e)
