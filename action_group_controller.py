#!/usr/bin/env python3
# encoding: utf-8
import os
import time
# import rospy  # Only needed if use_ros=True
# import sqlite3 as sql  # Only needed for action group files
from bus_servo_control import BusServoControl
# from ros_robot_controller.msg import SetBusServoState, BusServoState  # Only needed if use_ros=True

class ActionGroupController():
    runningAction = False
    stopRunning = False

    action_path = os.path.split(os.path.realpath(__file__))[0]

    def __init__(self, board=None, use_ros=False):
        # Minimal stub for direct servo test (no ROS, no DB)
        self.board = BusServoControl(board)
        self.use_ros = use_ros

    def stop_servo(self):
        self.board.stopBusServo([1, 2, 3, 4, 5, 6])

    def stop_action_group(self):
        self.stopRunning = True

    def runAction(self, actNum):
        print(f"Stub: runAction({actNum}) called (not implemented in minimal test mode)")
