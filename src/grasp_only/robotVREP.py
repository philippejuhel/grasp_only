# coding: utf-8

import time
import numpy as np
import grasp_only.vrep as vrep
from grasp_only.srv import GripperCmd, GripperCmdResponse, CoordAction, CoordActionResponse, CoordActionRequest
import rospy
from std_srvs.srv import Empty,EmptyResponse


class RobotVREP(object):
    def __init__(self, workspace_limits, ip_vrep):
        self.workspace_limits = workspace_limits
        s = rospy.Service('cmd_gripper', GripperCmd, self.cmdGripper)
        s = rospy.Service('robot_grasp', CoordAction, self.grasp)
        s = rospy.Service('robot_push', CoordAction, self.push)
        s = rospy.Service('restart_sim', Empty, self.restart_sim)
        # Make sure to have the server side running in V-REP:
        # in a child script of a V-REP scene, add following command
        # to be executed just once, at simulation start:
        #
        # simExtRemoteApiStart(19999)
        #
        # then start simulation, and run this program.
        #
        # IMPORTANT: for each successful call to simxStart, there
        # should be a corresponding call to simxFinish at the end!

        # MODIFY remoteApiConnections.txt

        # Connect to simulator
        self.sim_client = vrep.simxStart(ip_vrep, 20003, True, True, 5000, 5)  # Connect to V-REP on port 20003
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation on port {}'.format(self.sim_client))
        # Start simulation
        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target', vrep.simx_opmode_blocking)
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5, 0, 0.3), vrep.simx_opmode_blocking)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4:  # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)

    def close_gripper(self, async=False):
        """
        Close the gripper
        :param async:
        :return:
        """
        gripper_motor_velocity = -0.5
        gripper_motor_force = 100
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)

    def gripper_fully_closed(self):
        """
        Test if the gripper is closed
        :return: True if the gripper is fully closed (so, no object is holded)
        """
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
        while gripper_joint_position > -0.044:  # Block until gripper is fully closed
            sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            if new_gripper_joint_position >= gripper_joint_position:
                return False
            gripper_joint_position = new_gripper_joint_position
        return True

    def open_gripper(self, async=False):
        gripper_motor_velocity = 0.5
        gripper_motor_force = 20
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
    # ICAM            while gripper_joint_position < 0.0536: # Block until gripper is fully open
    #                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)

    def move_to(self, tool_position, tool_orientation):
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.02 * move_direction / move_magnitude
        num_move_steps = int(np.floor(move_magnitude / 0.02))
        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                       (UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),
                                       vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)

    def compute_nb_steps(self, move_direction, move_step):
        """
        Compute the number of steps for a non null coordinate (to correct the bug when move_step[0] when the robot is just above the piece to grasp)
        :param move_direction:
        :param move_step:
        :return:
        """
        if move_step[0] != 0:
            return int(np.floor(move_direction[0] / move_step[0]))
        elif move_step[1] != 0:
            return int(np.floor(move_direction[1] / move_step[1]))
        elif move_step[2] != 0:
            return int(np.floor(move_direction[2] / move_step[2]))
        else:
            print("Vector {} is null in grasp")

    # Gestion des services

    def restart_sim(self,req):
        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4: # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        return EmptyResponse()

    def cmdGripper(self, req):
        """
        Appelée par le service cmd_gripper
        :param req: True : ouvre la pince, False : ferme la pince
        :return:
        """
        if req.open:
            self.open_gripper()
        else:
            self.close_gripper()
        return GripperCmdResponse()

    def grasp(self, req):
        """
        Appelée par le service robot_grasp
        Try to grasp an object which position and orientation is given by the CoordAction message parameter
        :param req: a CoordAction parameter
        :return:
        """
        try:
            heightmap_rotation_angle = req.angle
            position = req.position
            print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))
            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2
            # Avoid collision with floor
            position = np.asarray(position).copy()
            position[2] = max(position[2] - 0.04, self.workspace_limits[2][0] + 0.02)
            # Move gripper to location above grasp target
            grasp_location_margin = 0.15
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)
            # Compute gripper position and linear movement increments
            tool_position = location_above_grasp_target
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05 * move_direction / move_magnitude
            num_move_steps = self.compute_nb_steps(move_direction, move_step)
            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))
            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps), UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                              (np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)
            # Ensure gripper is open
            self.open_gripper()
            # Approach grasp target
            self.move_to(position, None)
            # Close gripper to grasp target
            self.close_gripper()
            # Move gripper to location above grasp target
            self.move_to(location_above_grasp_target, None)
            # Check if grasp is successful
            gripper_full_closed = self.gripper_fully_closed()
            grasp_success = not gripper_full_closed
            # Move the grasped object elsewhere
            if grasp_success:
                rospy.wait_for_service('robot_push')
                try:
                    cmdMove = rospy.ServiceProxy('move_object', Empty)
                    resp = cmdMove()
                except rospy.ServiceException as e:
                    print("Service move_object call failed: %s" % e)
            return CoordActionResponse(grasp_success)
        except Exception as ex:  # Pour traquer une erreur "cannot convert float Nan to integer"
            print(ex)
            print("--------------> pb dans GRASP")
            print(move_direction[0])
            print(move_step[0])
            return CoordActionResponse(False)

    def push(self, req):
        """
        Appelée par le service robot_push
        Try to push an object which position and orientation is given by the CoordAction message parameter
        :param req: a CoordAction message
        :return:
        """
        try:
            heightmap_rotation_angle = req.angle
            position = list(req.position)
            print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))
            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2
            # Adjust pushing point to be on tip of finger
            position[2] = position[2] + 0.026
            # Compute pushing direction
            push_orientation = [1.0, 0.0]
            push_direction = np.asarray([push_orientation[0] * np.cos(heightmap_rotation_angle) - push_orientation[1] * np.sin(heightmap_rotation_angle),
                                         push_orientation[0] * np.sin(heightmap_rotation_angle) + push_orientation[1] * np.cos(heightmap_rotation_angle)])
            # Move gripper to location above pushing point
            pushing_point_margin = 0.1
            location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)
            # Compute gripper position and linear movement increments
            tool_position = location_above_pushing_point
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05 * move_direction / move_magnitude
            num_move_steps = self.compute_nb_steps(move_direction, move_step)
            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))
            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps), UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                              (np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)
            # Ensure gripper is closed
            self.close_gripper()
            # Approach pushing point
            self.move_to(position, None)
            # Compute target location (push to the right)
            push_length = 0.1
            target_x = min(max(position[0] + push_direction[0] * push_length, self.workspace_limits[0][0]), self.workspace_limits[0][1])
            target_y = min(max(position[1] + push_direction[1] * push_length, self.workspace_limits[1][0]), self.workspace_limits[1][1])
            push_length = np.sqrt(np.power(target_x - position[0], 2) + np.power(target_y - position[1], 2))
            # Move in pushing direction towards target location
            self.move_to([target_x, target_y, position[2]], None)
            # Move gripper to location above grasp target
            self.move_to([target_x, target_y, location_above_pushing_point[2]], None)
            return CoordActionResponse(True)
        except:  # Pour traquer une erreur "cannot convert float Nan to integer"
            print("--------------> pb dans PUSH")
            print(move_direction[0])
            print(move_step[0])
            return CoordActionResponse(False)


#
# Programme principal
#
if __name__ == '__main__':
    print("robot at home")
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
    ipVREP = '127.0.0.1'
    monRobot = RobotVREP(workspace_limits, ipVREP)
    monRobot.open_gripper()
    time.sleep(1)
    # print(monRobot.check_grasp())
    print('on ferme la pince')
    monRobot.close_gripper()
    # print(monRobot.check_grasp())
    print('push')
    monRobot.push(CoordActionRequest([-0.65, 0.1, 0.1], 0.1))
    print('grasp')
    monRobot.grasp(CoordActionRequest([-0.5, 0.1, 0.1], 0.1))
    print('fin du programme de test')
