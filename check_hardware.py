#!/usr/bin/env python3
"""
Hardware Verification Script — Identify Why Arm Isn't Moving
==============================================================

Checks:
1. I2C device responds
2. I2C command format is correct
3. Returns data from servo MCU (should indicate servo state)
"""

import time
import logging
from smbus2 import SMBus, i2c_msg

logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_I2C_BUS = 1
_I2C_ADDR = 0x34
_I2C_REG_BUS = 21

def test_i2c_raw():
    """Test raw I2C communication with the servo MCU."""
    logger.info("=" * 60)
    logger.info("RAW I2C COMMUNICATION TEST")
    logger.info("=" * 60)
    
    try:
        with SMBus(_I2C_BUS) as bus:
            logger.info(f"I2C Bus {_I2C_BUS} opened")
            
            # Test 1: Read from device
            logger.info("\n[1] Reading from 0x34...")
            try:
                val = bus.read_byte_data(_I2C_ADDR, 0)
                logger.info(f"    ✓ Read: 0x{val:02X} (decimal: {val})")
            except Exception as e:
                logger.error(f"    ✗ Read failed: {e}")
                return False
            
            # Test 2: Write a command to servo 5
            logger.info("\n[2] Sending I2C command to move servo 5...")
            try:
                # Format: [register, 1, duration_lo, duration_hi, servo_id, pulse_lo, pulse_hi]
                # duration: 1000ms = 0x03E8
                # pulse: 400 = 0x0190
                buf = [_I2C_REG_BUS, 1, 0xE8, 0x03, 5, 0x90, 0x01]
                msg = i2c_msg.write(_I2C_ADDR, buf)
                result = bus.i2c_rdwr(msg)
                logger.info(f"    ✓ Write command sent: {buf}")
                logger.info(f"    ✓ Result: {result}")
            except Exception as e:
                logger.error(f"    ✗ Write failed: {e}")
                return False
            
            # Wait a bit for servo to move
            time.sleep(1.2)
            
            # Test 3: Read servo position/status
            logger.info("\n[3] Reading servo feedback from 0x34...")
            try:
                val = bus.read_byte_data(_I2C_ADDR, 0)
                logger.info(f"    ✓ Feedback: 0x{val:02X} (decimal: {val})")
                logger.info("    (If 0x77, servo MCU responded successfully)")
            except Exception as e:
                logger.error(f"    ✗ Read failed: {e}")
            
            # Test 4: List register addresses
            logger.info("\n[4] Checking other registers...")
            for reg in [0, 1, 20, 21, 40, 41]:
                try:
                    val = bus.read_byte_data(_I2C_ADDR, reg)
                    logger.info(f"    Reg {reg:3d} (0x{reg:02X}) = 0x{val:02X}")
                except:
                    logger.debug(f"    Reg {reg:3d} — not readable")
            
            return True
            
    except Exception as e:
        logger.error(f"✗ I2C test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_servo_power():
    """Check if servos are drawing power (via LED indicator)."""
    logger.info("\n" + "=" * 60)
    logger.info("SERVO POWER CHECK")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Please check physically:")
    logger.info("  1. Does the servo MCU have a power LED? (should be lit)")
    logger.info("  2. Do the servo motors have a power indicator? (LED or lights)")
    logger.info("  3. Do servo motors make a sound/chirp when powered on?")
    logger.info("")
    
    response = input("Enter 'yes' if you see servo power indicators, 'no' if not: ").lower()
    return response.startswith('y')


def diagnose():
    """Run complete diagnosis."""
    logger.info("\n")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " " * 8 + "HARDWARE VERIFICATION & DIAGNOSIS" + " " * 17 + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")
    
    i2c_ok = test_i2c_raw()
    power_ok = check_servo_power()
    
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSIS RESULTS")
    logger.info("=" * 60)
    
    if i2c_ok and power_ok:
        logger.info("✅ I2C communication: WORKING")
        logger.info("✅ Servo power: CONFIRMED")
        logger.info("")
        logger.info("Servos should be responding. If arm still doesn't move:")
        logger.info("  → Check servo cable connections to servo MCU")
        logger.info("  → Verify servo motor connectors are seated properly")
        logger.info("  → Try manual servo rotation (should feel resistance)")
        return "hardware_connected_check_cables"
    
    elif i2c_ok and not power_ok:
        logger.error("✅ I2C communication: WORKING")
        logger.error("❌ Servo power: NOT DETECTED")
        logger.error("")
        logger.error("Problem: Servo motors are not powered")
        logger.error("  → Check servo power supply (usually needs 12V)")
        logger.error("  → Verify power connections to servo board")
        logger.error("  → Check fuses on servo board (if present)")
        return "no_servo_power"
    
    elif not i2c_ok:
        logger.error("❌ I2C communication: FAILED")
        logger.error("")
        logger.error("Problem: Cannot communicate with servo MCU at 0x34")
        logger.error("  → Is the servo MCU actually powered?")
        logger.error("  → Is it connected to I2C bus 1?")
        logger.error("  → Try: sudo i2cdetect -y 1 (from terminal)")
        logger.error("  → Look for device at address 34 in the output")
        return "i2c_communication_failed"
    
    else:
        logger.error("❌ Multiple issues detected")
        return "multiple_issues"


if __name__ == "__main__":
    result = diagnose()
    
    logger.info("\n" + "=" * 60)
    logger.info(f"NEXT STEPS ({result})")
    logger.info("=" * 60)
    
    if result == "no_servo_power":
        logger.info("1. Power on the servo motor supply (typically 12V)")
        logger.info("2. Run test again: python test_servo_communication.py")
    
    elif result == "i2c_communication_failed":
        logger.info("1. Check that servo MCU is powered")
        logger.info("2. From terminal: sudo i2cdetect -y 1")
        logger.info("3. Look for device at address 0x34")
    
    elif result == "hardware_connected_check_cables":
        logger.info("1. Visually inspect all servo cable connections")
        logger.info("2. Try gently rotating a servo motor by hand")
        logger.info("   (should have resistance, not spin freely)")
        logger.info("3. Try moving arm again: python main.py --no-cut --single-pass")
    
    logger.info("")
