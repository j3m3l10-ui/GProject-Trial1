#!/usr/bin/env python3
import sys
import time
from ros_robot_controller_sdk import *

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

class BusServoControl:
    def __init__(self, Board, time_out=100):
        self.board = Board
        self.time_out = time_out

    def setBusServoID(self, oldid, newid):
        self.board.bus_servo_set_id(oldid, newid)

    def getBusServoID(self, servo_id=None):
        count = 0 
        while True:
            if servo_id is None:
                res = self.board.bus_servo_read_id()
            else:
                res = self.board.bus_servo_read_id(servo_id)
            count += 1
            if res is not None:
                return res
            if count > self.time_out:
                return None
            time.sleep(0.01)

    def setBusServoPulse(self, servo_id, position, duration):
        position = 0 if position < 0 else position
        position = 1000 if position > 1000 else position
        duration = 0 if duration < 0 else duration
        duration = 30000 if duration > 30000 else duration
        self.board.bus_servo_set_position(duration/1000, [[servo_id, position]])

    def getBusServoPulse(self, servo_id):
        count = 0 
        while True:
            res = self.board.bus_servo_read_position(servo_id)
            count += 1
            if res is not None:
                return res
            if count > self.time_out:
                return None
            time.sleep(0.01)

    def stopBusServo(self, servo_id):
        self.board.bus_servo_stop(servo_id)

    def setBusServoDeviation(self, servo_id, offset=0):
        self.board.bus_servo_set_offset(servo_id, offset)

    def saveBusServoDeviation(self, servo_id):
        self.board.bus_servo_save_offset(servo_id)

    def getBusServoDeviation(self, servo_id):
        count = 0
        while True:
            res = self.board.bus_servo_read_offset(servo_id)
            count += 1
            if res is not None:
                return res
            if count > self.time_out:
                return None
            time.sleep(0.01)

    def setBusServoAngleLimit(self, servo_id, low, high):
        self.board.bus_servo_set_angle_limit(servo_id, [low, high])

    def getBusServoAngleLimit(self, servo_id):
        count = 0
        while True:
            res = self.board.bus_servo_read_angle_limit(servo_id)
            count += 1
            if res is not None:
                return res
            if count > self.time_out:
                return None
            time.sleep(0.01)

    def setBusServoVinLimit(self, servo_id, low, high):
        self.board.bus_servo_set_vin_limit(servo_id, [low, high])

    def getBusServoVinLimit(self, servo_id):
        count = 0
        while True:
            res = self.board.bus_servo_read_vin_limit(servo_id)
            count += 1
            if res is not None:
                return res
            if count > self.time_out:
                return None
            time.sleep(0.01)

    def setBusServoMaxTemp(self, servo_id, m_temp):
        self.board.bus_servo_set_temp_limit(servo_id, m_temp)

    def getBusServoTempLimit(self, servo_id):
        count = 0
        while True:
            res = self.board.bus_servo_read_temp_limit(servo_id)
            count += 1
            if res is not None:
                return res
            if count > self.time_out:
                return None
            time.sleep(0.01)

    def getBusServoTemp(self, servo_id):
        count = 0
        while True:
            res = self.board.bus_servo_read_temp(servo_id)
            count += 1
            if res is not None:
                return res
            if count > self.time_out:
                return None
            time.sleep(0.01)

    def getBusServoVin(self, servo_id):
        count = 0
        while True:
            res = self.board.bus_servo_read_vin(servo_id)
            count += 1
            if res is not None:
                return res
            if count > self.time_out:
                return None
            time.sleep(0.01)

    def restBusServoPulse(self, oldid):
        setBusServoDeviation(oldid, 0)
        time.sleep(0.1)
        setBusServoPulse(oldid, 500, 100)

    def unloadBusServo(self, servo_id):
        self.board.bus_servo_enable_torque(servo_id, 1)

    def getBusServoLoadStatus(self, servo_id):
        count = 0
        while True:
            res = self.board.bus_servo_read_torque_state(servo_id)
            count += 1
            if res is not None:
                return res
            if count > self.time_out:
                return None
            time.sleep(0.01)
