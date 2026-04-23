#!/usr/bin/env python3
"""
Test Servo Communication — Verify I2C commands reach the arm
=============================================================

Usage:
    python test_servo_communication.py
"""

import time
import logging

logging.basicConfig(level=logging.DEBUG,
                   format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_i2c_device():
    """Check if I2C device at 0x34 is accessible."""
    logger.info("=" * 60)
    logger.info("TESTING I2C DEVICE AT 0x34...")
    logger.info("=" * 60)
    
    try:
        from smbus2 import SMBus
        with SMBus(1) as bus:
            # Try to read from I2C device
            try:
                data = bus.read_byte_data(0x34, 0)
                logger.info(f"✓ I2C device at 0x34 responded with byte: 0x{data:02X}")
                return True
            except Exception as e:
                logger.error(f"✗ I2C device at 0x34 not responding: {e}")
                return False
    except Exception as e:
        logger.error(f"✗ I2C bus error: {e}")
        return False


def test_servo_driver_direct():
    """Test servo driver directly."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING SERVO DRIVER IN REAL MODE...")
    logger.info("=" * 60)
    
    try:
        from servo_driver import ServoDriver
        
        driver = ServoDriver(mode="real")
        logger.info(f"Backend: {driver._backend}")
        
        if driver._backend == "sim":
            logger.warning("✗ Servo driver fell back to SIM mode!")
            logger.warning("  This means I2C/UART communication failed.")
            return False
        
        logger.info("✓ Servo driver initialized in real mode")
        
        # Try moving servo 5 (shoulder) — slow, safe move
        logger.info("\nAttempting to move servo 5 (shoulder)...")
        logger.info("Watch the arm shoulder joint for movement...")
        
        driver.move_servo(5, 400, duration_ms=1000)  # Move left
        logger.info("Move command sent to servo 5 → position 400 (1000ms)")
        time.sleep(1.5)
        
        driver.move_servo(5, 600, duration_ms=1000)  # Move right
        logger.info("Move command sent to servo 5 → position 600 (1000ms)")
        time.sleep(1.5)
        
        driver.move_servo(5, 500, duration_ms=1000)  # Center
        logger.info("Move command sent to servo 5 → position 500 (1000ms)")
        time.sleep(1.5)
        
        logger.info("✓ Move commands completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Servo driver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_arm_movement():
    """Test moving the actual arm."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING ARM CONTROLLER...")
    logger.info("=" * 60)
    
    try:
        from arm_controller import FiveDOFArm
        
        arm = FiveDOFArm()
        logger.info(f"Current arm angles: {arm.joint_angles}")
        
        # Try a small angle change
        initial_angles = arm.joint_angles.copy()
        logger.info(f"Initial: {initial_angles}")
        
        new_angles = initial_angles.copy()
        new_angles[4] = initial_angles[4] + 0.2  # Small base rotation
        
        logger.info(f"Setting arm angles to: {new_angles}")
        arm.set_joint_angles(new_angles)
        
        logger.info(f"New arm angles: {arm.joint_angles}")
        
        if arm.joint_angles[4] != new_angles[4]:
            logger.warning("⚠ Arm angles changed but different from commanded!")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Arm test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_movement():
    """Test full arm movement to a target."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING FULL ARM MOVEMENT...")
    logger.info("=" * 60)
    
    try:
        from arm_controller import FiveDOFArm, angles_to_pulses
        from servo_driver import ServoDriver
        
        arm = FiveDOFArm()
        driver = ServoDriver(mode="real")
        
        logger.info("Current arm angles: " + str(arm.joint_angles))
        
        # Set a simple target
        target = arm.joint_angles.copy()
        target[0] += 0.3  # Small movement on first joint
        
        logger.info(f"Target angles: {target}")
        
        arm.set_joint_angles(target)
        pulses = angles_to_pulses(target)
        
        logger.info(f"Target pulses: {pulses}")
        logger.info("Sending move commands...")
        
        driver.move_servos(pulses, duration_ms=1000)
        
        logger.info("Move sent! Waiting 1.5 seconds...")
        time.sleep(1.5)
        
        logger.info(f"Final arm angles: {arm.joint_angles}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Full movement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("\n")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " " * 10 + "SERVO COMMUNICATION DIAGNOSTIC TEST" + " " * 14 + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")
    
    results = {}
    
    results["i2c"] = test_i2c_device()
    results["driver"] = test_servo_driver_direct()
    results["arm"] = test_arm_movement()
    results["full"] = test_full_movement()
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{name.upper():12} → {status}")
    
    logger.info("")
    
    all_pass = all(results.values())
    if all_pass:
        logger.info("✅ All tests passed! Arm should be moving.")
    else:
        logger.error("❌ Some tests failed. Check hardware connections:")
        logger.error("   - Is the servo MCU powered and connected to 0x34?")
        logger.error("   - Are the servo motors powered?")
        logger.error("   - Is the I2C bus enabled on the RPi?")
