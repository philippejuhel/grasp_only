#!/usr/bin/env python3
# coding: utf-8

import rospy
from grasp_only.simulation import Simulation
import argparse

#
# rosrun vpg_ros node_vpg.py --explore_rate_decay --save_visualizations
#


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')
    # --------------- Setup options ---------------
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store',default=0.002, help='meters per pixel of heightmap')
    # ------------- Algorithm options -------------
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store',default=0.5)
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    # ------ Pre-loading and logging options ------
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,help='save visualizations of FCN predictions?')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10, help='number of objects to add to simulation')
    parser.add_argument('--logging_directory', dest='logging_directory',action='store', default='../logs', help='directory for logging files')

    # Run main program with specified arguments
    args = parser.parse_args()
    simu = Simulation(args)
    simu.execute()
