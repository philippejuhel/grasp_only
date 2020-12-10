#!/usr/bin/env python3
# coding: utf-8

import rospy
import numpy as np
from grasp_only.cameraVREP import CameraVREP
from grasp_only.objectsVREP import ObjectsVREP
from grasp_only.robotVREP import RobotVREP


if __name__ == "__main__":
        rospy.init_node('node_environment_server')
        try:
                print(rospy.get_param_names())
        except Exception:
                print("could not get param name")
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
        ipVREP = '127.0.0.1'
        myRobot = RobotVREP(workspace_limits,ipVREP)
        print("Ready RobotVREP")
        objects_dir = rospy.get_param("/node_environment_server/objects_dir")
        myObjects = ObjectsVREP(workspace_limits, objects_dir)  # Se mettre dans le répertoire catkin_ws
        print("Ready ObjetsVREP")
        myCamera = CameraVREP()
        print("Ready CameraVREP")
        rospy.spin()

