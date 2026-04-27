#!/usr/bin/env python3
# encoding: utf-8
import os
import time
import sqlite3 as sql
from bus_servo_control import BusServoControl

class ActionGroupController():
    runningAction = False
    stopRunning = False

    action_path = os.path.split(os.path.realpath(__file__))[0]

    def __init__(self, board=None, use_ros=False):
        self.board = BusServoControl(board)
        self.use_ros = use_ros

    def stop_servo(self):
        self.board.stopBusServo([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

    def stop_action_group(self):
        self.stopRunning = True

    def runAction(self, actNum):
        '''
        Run an action group from a .d6a SQLite file.
        :param actNum: action group name (str) or full path to .d6a file
        '''
        if actNum is None:
            return

        # Resolve path — check working dir, script dir, and ActionGroups subdir
        name = str(actNum).strip()
        if not name.endswith('.d6a'):
            name += '.d6a'

        candidates = [
            name,
            os.path.join(self.action_path, name),
            os.path.join(self.action_path, 'ActionGroups', name),
        ]
        actPath = None
        for c in candidates:
            if os.path.isfile(c):
                actPath = c
                break

        if actPath is None:
            self.runningAction = False
            print(f"未能找到动作组文件: {name}")
            return

        self.stopRunning = False
        if self.runningAction:
            return

        self.runningAction = True
        ag = sql.connect(actPath)
        cu = ag.cursor()
        cu.execute("select * from ActionGroup")
        while True:
            act = cu.fetchone()
            if self.stopRunning is True:
                self.stopRunning = False
                break
            if act is not None:
                # act[0]=Index, act[1]=Time, act[2..]=Servo1..N
                for i in range(0, len(act) - 2, 1):
                    servo_id = i + 1
                    # Skip Servo2 (ID2 not present on this arm)
                    if servo_id == 2:
                        continue
                    self.board.setBusServoPulse(servo_id, act[2 + i], act[1])
                time.sleep(float(act[1]) / 1000.0)
            else:
                break
        self.runningAction = False
        cu.close()
        ag.close()
