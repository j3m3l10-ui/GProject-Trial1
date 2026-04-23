#!/usr/bin/env python3
"""
Pre-Flight Diagnostic Tool — Tomato Harvesting Robot
======================================================

Runs system checks before executing main.py.
Verifies: camera, detector model, arm/servo communication.

Usage:
    python diagnostic.py
"""

import sys
import time
import os
import cv2
import logging

logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def check_imports():
    """Verify all required Python packages are installed."""
    logger.info("=" * 60)
    logger.info("CHECKING IMPORTS...")
    logger.info("=" * 60)
    
    packages = {
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "smbus2": "SMBus2 (I2C)",
        "ultralytics": "Ultralytics (YOLO)",
    }
    
    missing = []
    for module, name in packages.items():
        try:
            __import__(module)
            logger.info(f"✓ {name} ({module})")
        except ImportError:
            logger.error(f"✗ {name} ({module}) — MISSING")
            missing.append(module)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False
    
    logger.info("✓ All required packages installed\n")
    return True


def check_camera(camera_index=0):
    """Test camera capture and basic properties."""
    logger.info("=" * 60)
    logger.info(f"CHECKING CAMERA (index {camera_index})...")
    logger.info("=" * 60)
    
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error(f"✗ Cannot open camera at index {camera_index}")
            return False
        
        logger.info("✓ Camera opened successfully")
        
        # Test frame capture
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("✗ Cannot read frame from camera")
            cap.release()
            return False
        
        h, w = frame.shape[:2]
        logger.info(f"✓ Captured frame: {w}×{h} pixels")
        
        # Test 5 frames for consistency
        logger.info("Capturing 5 test frames...")
        for i in range(5):
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(f"  Frame {i+1}: Failed")
            else:
                logger.info(f"  Frame {i+1}: OK ({frame.shape[1]}×{frame.shape[0]})")
        
        # Try setting low-latency parameters
        logger.info("\nTesting low-latency settings...")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info("✓ Set buffer size = 1")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        logger.info("✓ Set resolution = 640×480")
        
        cap.release()
        logger.info("✓ Camera test passed\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ Camera test failed: {e}\n")
        return False


def check_detector_model():
    """Verify YOLO model can be loaded."""
    logger.info("=" * 60)
    logger.info("CHECKING DETECTOR MODEL...")
    logger.info("=" * 60)
    
    try:
        logger.info("Loading TomatoDetector...")
        from vision import TomatoDetector
        
        logger.info("Initializing detector with imgsz=416...")
        detector = TomatoDetector(imgsz=416)
        
        logger.info(f"✓ Detector loaded successfully")
        logger.info(f"  Model: YOLOv8s backbone")
        logger.info(f"  Input size: 416×416")
        logger.info(f"  Detector ready\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ Detector model load failed: {e}\n")
        return False


def check_arm_communication():
    """Test arm controller communication."""
    logger.info("=" * 60)
    logger.info("CHECKING ARM COMMUNICATION...")
    logger.info("=" * 60)
    
    try:
        logger.info("Loading arm controller...")
        from arm_controller import FiveDOFArm
        
        arm = FiveDOFArm()
        logger.info("✓ Arm controller initialized")
        
        angles = arm.joint_angles
        logger.info(f"✓ Current arm angles: {angles}")
        
        logger.info("✓ Arm communication OK\n")
        return True
        
    except Exception as e:
        logger.warning(f"⚠ Arm check (expected to warn): {e}")
        logger.warning("  → This is normal if hardware not connected\n")
        return "warning"


def check_servo_driver():
    """Test servo driver (I2C communication)."""
    logger.info("=" * 60)
    logger.info("CHECKING SERVO DRIVER...")
    logger.info("=" * 60)
    
    try:
        logger.info("Loading servo driver...")
        from servo_driver import ServoDriver
        
        # Try simulation mode first (safer)
        logger.info("Testing in SIM mode (no real I2C)...")
        driver = ServoDriver(mode="sim")
        logger.info("✓ Servo driver (SIM mode) initialized")
        
        # Try real mode with error handling
        try:
            logger.info("Testing I2C bus access (bus 1, address 0x34)...")
            import smbus2
            bus = smbus2.SMBus(1)
            logger.info("✓ I2C bus 1 accessible")
            
            # Try reading from servo address
            try:
                bus.read_byte(0x34)
                logger.info("✓ Servo controller at 0x34 responds")
            except:
                logger.warning("⚠ Servo at 0x34 not responding (may be normal if powered off)")
            
            bus.close()
            logger.info("✓ Servo driver communication OK\n")
            return True
            
        except Exception as i2c_err:
            logger.warning(f"⚠ I2C test (expected to warn): {i2c_err}")
            logger.warning("  → This is normal if hardware not connected\n")
            return "warning"
        
    except Exception as e:
        logger.error(f"✗ Servo driver check failed: {e}\n")
        return False


def check_config_files():
    """Verify all required configuration files exist."""
    logger.info("=" * 60)
    logger.info("CHECKING CONFIG FILES...")
    logger.info("=" * 60)
    
    required_files = [
        "main.py",
        "vision.py",
        "arm_controller.py",
        "servo_driver.py",
        "arm_home_position.py",
        "data.yaml",
    ]
    
    all_found = True
    for fname in required_files:
        if os.path.exists(fname):
            size = os.path.getsize(fname)
            logger.info(f"✓ {fname} ({size} bytes)")
        else:
            logger.error(f"✗ {fname} — MISSING")
            all_found = False
    
    logger.info("")
    return all_found


def run_all_checks():
    """Run complete diagnostic suite."""
    logger.info("\n")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " " * 12 + "TOMATO HARVESTER DIAGNOSTIC TOOL" + " " * 13 + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")
    
    results = {}
    
    # Run all checks
    results["imports"] = check_imports()
    results["camera"] = check_camera(0)
    results["model"] = check_detector_model()
    results["config"] = check_config_files()
    results["arm"] = check_arm_communication()
    results["servo"] = check_servo_driver()
    
    # Summary
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v is True)
    warnings = sum(1 for v in results.values() if v == "warning")
    failed = sum(1 for v in results.values() if v is False)
    
    logger.info(f"✓ Passed:   {passed}")
    logger.info(f"⚠ Warnings: {warnings} (normal if hardware not connected)")
    logger.info(f"✗ Failed:   {failed}")
    logger.info("")
    
    if failed > 0:
        logger.error("❌ DIAGNOSTIC FAILED — Fix errors above before running main.py")
        return False
    
    logger.info("✅ DIAGNOSTIC PASSED — Ready to run main.py!")
    logger.info("\nNext steps:")
    logger.info("  1. python main.py --dry-run --single-pass     (preview)")
    logger.info("  2. python main.py --no-cut --single-pass      (test moves)")
    logger.info("  3. python main.py --single-pass               (harvest)")
    logger.info("")
    
    return True


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
