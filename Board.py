#!/usr/bin/env python3
# coding: utf-8 

import os
import sys
import time

# lgpio (used by RPi.GPIO on Pi 5) creates pipe files in CWD.
# FAT/exFAT filesystems (USB drives) don't support pipes, so we
# temporarily switch to /tmp for the import.
_orig_cwd = os.getcwd()
try:
    os.chdir("/tmp")
    import RPi.GPIO as GPIO
finally:
    os.chdir(_orig_cwd)

from smbus2 import SMBus, i2c_msg

# RGB libraries are commented out to prevent ModuleNotFoundError
# from rpi_ws281x import PixelStrip
# from rpi_ws281x import Color as PixelColor

# Import your custom/mocked yaml_handle
try:
    import yaml_handle
except ImportError:
    # Create a dummy class to prevent crashes if file is missing
    class DummyYaml:
        Deviation_file_path = ""
        def get_yaml_data(self, path):
            return {str(i): 0 for i in range(1, 7)}
    yaml_handle = DummyYaml()
    print("Warning: yaml_handle.py not found. Using default zero offsets.")

# Hardware Addresses
__ADC_BAT_ADDR = 0
__SERVO_ADDR   = 21
__MOTOR_ADDR   = 31
__SERVO_ADDR_CMD  = 40
# Default to the common Hiwonder motor controller address seen on i2cdetect.
# Can be overridden at runtime, e.g. HIWONDER_I2C_ADDR=0x34 HIWONDER_I2C_BUS=1
__i2c_addr = int(os.environ.get('HIWONDER_I2C_ADDR', '0x34'), 0)
__i2c = int(os.environ.get('HIWONDER_I2C_BUS', '1'), 0)

# State Tracking
__motor_speed = [0, 0, 0, 0]
__servo_angle = [0, 0, 0, 0, 0, 0]
__servo_pulse = [0, 0, 0, 0, 0, 0]

# GPIO Setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# --- RGB LED Configuration Neutralized ---
# We keep these variables so other functions don't break, 
# but we disable the hardware initialization.
class MockRGB:
    def begin(self): pass
    def show(self): pass
    def setPixelColor(self, i, color): pass
    def numPixels(self): return 0

RGB = MockRGB() 

def setMotor(index, speed):
    """Sets the speed for a specific motor (1-4)"""
    if index < 1 or index > 4:
        raise AttributeError("Invalid motor num: %d" % index)
    
    # Mirror logic for chassis symmetry
    # Motors on one side need to be inverted to move forward together
    if index == 2 or index == 4:
        speed = speed
    else:
        speed = -speed
        
    index = index - 1
    # Constrain speed to -100 to 100 range
    speed = 100 if speed > 100 else speed
    speed = -100 if speed < -100 else speed
    reg = __MOTOR_ADDR + index
    
    with SMBus(__i2c) as bus:
        try:
            # Convert speed to signed byte for I2C transfer
            msg = i2c_msg.write(__i2c_addr, [reg, speed.to_bytes(1, 'little', signed=True)[0]])
            bus.i2c_rdwr(msg)
            __motor_speed[index] = speed
        except:
            # Retry once on failure
            msg = i2c_msg.write(__i2c_addr, [reg, speed.to_bytes(1, 'little', signed=True)[0]])
            bus.i2c_rdwr(msg)
            __motor_speed[index] = speed
           
    return __motor_speed[index]

def setBusServoPulse(servo_id, pulse=500, use_time=500):
    """Controls serial bus servos (LX-16A / LX-225) via I2C relay.

    The expansion board MCU converts this I2C command into a serial bus
    servo packet on the half-duplex UART connected to the servo daisy chain.

    Args:
        servo_id: Bus servo ID (1–254, 254=broadcast)
        pulse:    Target position 0–1000 (centre ≈ 500)
        use_time: Move duration in milliseconds (0–30000)
    """
    if servo_id < 1 or servo_id > 254:
        raise AttributeError("Invalid Servo ID: %d" % servo_id)

    deviation_data = yaml_handle.get_yaml_data(yaml_handle.Deviation_file_path)
    pulse += deviation_data.get(str(servo_id), 0)
    pulse = max(0, min(1000, int(pulse)))
    use_time = max(0, min(30000, int(use_time)))

    index = servo_id - 1
    buf = [__SERVO_ADDR, 1] + list(use_time.to_bytes(2, 'little')) + \
          [servo_id] + list(pulse.to_bytes(2, 'little'))

    with SMBus(__i2c) as bus:
        try:
            msg = i2c_msg.write(__i2c_addr, buf)
            bus.i2c_rdwr(msg)
            if index < 6:
                __servo_pulse[index] = pulse
                __servo_angle[index] = int(pulse * 0.24)
        except Exception as e:
            msg = i2c_msg.write(__i2c_addr, buf)
            bus.i2c_rdwr(msg)

    return pulse


def setPWMServoPulse(servo_id, pulse=1500, use_time=1000):
    """Controls the PWM servos (usually for the arm)"""
    if servo_id < 1 or servo_id > 6:
        raise AttributeError("Invalid Servo ID: %d" % servo_id)
    
    # Fetch offset data
    deviation_data = yaml_handle.get_yaml_data(yaml_handle.Deviation_file_path)
    index = servo_id - 1
    
    # Apply offset and constrain pulse (500-2500ms)
    pulse += deviation_data.get(str(servo_id), 0)
    pulse = 500 if pulse < 500 else pulse
    pulse = 2500 if pulse > 2500 else pulse
    
    # FIX: Ensure this line is closed properly
    use_time = max(0, min(30000, use_time))
    
    # Package command for I2C
    buf = [__SERVO_ADDR_CMD, 1] + list(use_time.to_bytes(2, 'little')) + [servo_id,] + list(pulse.to_bytes(2, 'little'))
    
    with SMBus(__i2c) as bus:
        try:
            msg = i2c_msg.write(__i2c_addr, buf)
            bus.i2c_rdwr(msg)
            __servo_pulse[index] = pulse
            __servo_angle[index] = int((pulse - 500) * 0.09)
        except Exception as e:
            msg = i2c_msg.write(__i2c_addr, buf)
            bus.i2c_rdwr(msg)

    return __servo_pulse[index]
