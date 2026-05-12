import sys
import types
import unittest

import numpy as np


try:
    import ultralytics  # noqa: F401
except ImportError:
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=object)

import main
from arm_controller import CUT_GAP_M, compute_cut_point
from vision import TomatoDetector


def detection(x, y, z, confidence=0.8):
    return {
        "confidence": confidence,
        "xyz_cm": {"x": x, "y": y, "z": z},
    }


class FakeArm:
    def __init__(self, reachable=True, ik_result=(True, 0.0, 1)):
        self.base_pos = np.zeros(3, dtype=float)
        self.joint_angles = np.zeros(5, dtype=float)
        self.reachable = reachable
        self.ik_result = ik_result
        self.reach_checks = []
        self.set_angles_calls = []

    def is_reachable(self, target):
        self.reach_checks.append(np.array(target, dtype=float))
        return self.reachable

    def max_reach(self):
        return 0.388

    def inverse_kinematics(self, target, max_iters=300, tol=5e-4):
        self.ik_target = np.array(target, dtype=float)
        return self.ik_result

    def set_joint_angles(self, angles):
        self.joint_angles = np.array(angles, dtype=float)
        self.set_angles_calls.append(self.joint_angles.copy())


class FakeDriver:
    def __init__(self):
        self.calls = []

    def gripper_open(self, duration_ms=400):
        self.calls.append(("open", duration_ms))

    def gripper_close(self, duration_ms=400):
        self.calls.append(("close", duration_ms))

    def move_servos(self, pulses, duration_ms=500):
        self.calls.append(("move", dict(pulses), duration_ms))


class HarvestPipelineTest(unittest.TestCase):
    def setUp(self):
        self._sleep = main.time.sleep
        main.time.sleep = lambda _seconds: None

    def tearDown(self):
        main.time.sleep = self._sleep

    def test_cut_point_is_one_centimetre_above_tomato_surface(self):
        center = np.array([0.2, -0.05, 0.15], dtype=float)
        radius = 0.035
        edge, cut = compute_cut_point(center, radius, stem_direction=[0, 0, 2])

        np.testing.assert_allclose(edge, center + [0.0, 0.0, radius])
        np.testing.assert_allclose(cut, center + [0.0, 0.0, radius + CUT_GAP_M])

    def test_confirmation_tracks_the_same_tomato(self):
        buffer = [(0.0, detection(10.0, 0.0, 50.0, confidence=0.7))]

        near = detection(11.0, 0.5, 50.5, confidence=0.6)
        far = detection(35.0, 0.0, 80.0, confidence=0.95)
        selected, same_target = main._select_detection_for_confirmation([far, near], buffer)

        self.assertIs(selected, near)
        self.assertTrue(same_target)

        selected, same_target = main._select_detection_for_confirmation([far], buffer)
        self.assertIs(selected, far)
        self.assertFalse(same_target)

    def test_depth_uses_bbox_diameter_estimate(self):
        self.assertEqual(TomatoDetector.estimate_bbox_diameter_px(10, 20, 50, 80), 50.0)

    def test_harvest_reachability_uses_cut_point(self):
        arm = FakeArm(reachable=False)
        driver = FakeDriver()
        locked = {"x": 0.0, "y": 0.0, "z": 5.0}

        harvest_ok, reason = main._execute_harvest(arm, driver, locked, frame=None)

        target = main.camera_to_arm_frame(locked)
        _, expected_cut = compute_cut_point(
            target, main.TOMATO_RADIUS_M, arm.base_pos,
            stem_direction=main.STEM_DIRECTION_ARM_FRAME)
        np.testing.assert_allclose(arm.reach_checks[0], expected_cut)
        self.assertEqual(driver.calls, [])
        self.assertFalse(harvest_ok)
        self.assertIn("out of reach", reason)

    def test_harvest_aborts_when_ik_does_not_converge(self):
        arm = FakeArm(reachable=True, ik_result=(False, main.IK_MAX_ERROR_M / 2, 500))
        driver = FakeDriver()
        locked = {"x": 0.0, "y": 0.0, "z": 5.0}

        harvest_ok, reason = main._execute_harvest(arm, driver, locked, frame=None)

        self.assertIn(("open", 400), driver.calls)
        self.assertNotIn(("close", 500), driver.calls)
        self.assertFalse(any(call[0] == "move" for call in driver.calls))
        self.assertFalse(harvest_ok)
        self.assertIn("IK failed", reason)

    def test_confirmation_reaches_harvest_with_rolling_buffer(self):
        class Clock:
            def __init__(self):
                self.now = 0.0

            def time(self):
                return self.now

            def sleep(self, _seconds):
                pass

        class FakeDetector:
            def detect(self, frame):
                det = detection(0.0, 0.0, 10.0, confidence=0.9)
                det["bbox_px"] = [100, 100, 180, 180]
                det["center_px"] = [140, 140]
                return [det]

            def annotate(self, frame, detections):
                return frame.copy()

        class FakeCapture:
            def __init__(self, *_args, **_kwargs):
                pass

            def isOpened(self):
                return True

            def read(self):
                clock.now += 0.21
                return True, np.zeros((480, 640, 3), dtype=np.uint8)

            def release(self):
                pass

        clock = Clock()
        wait_calls = {"count": 0}
        harvest_calls = []
        originals = (
            main.TomatoDetector,
            main.cv2.VideoCapture,
            main.cv2.imshow,
            main.cv2.waitKey,
            main.cv2.destroyAllWindows,
            main.time.time,
            main.time.sleep,
            main._execute_harvest,
        )

        def fake_wait_key(_delay):
            wait_calls["count"] += 1
            return ord("q") if wait_calls["count"] >= 35 else -1

        def fake_execute(_arm, _driver, locked_xyz_cm, _frame):
            harvest_calls.append(dict(locked_xyz_cm))
            return True, "test harvest"

        try:
            main.TomatoDetector = FakeDetector
            main.cv2.VideoCapture = FakeCapture
            main.cv2.imshow = lambda *_args, **_kwargs: None
            main.cv2.waitKey = fake_wait_key
            main.cv2.destroyAllWindows = lambda: None
            main.time.time = clock.time
            main.time.sleep = clock.sleep
            main._execute_harvest = fake_execute

            main.run_harvesting(sim_mode=True)
        finally:
            (
                main.TomatoDetector,
                main.cv2.VideoCapture,
                main.cv2.imshow,
                main.cv2.waitKey,
                main.cv2.destroyAllWindows,
                main.time.time,
                main.time.sleep,
                main._execute_harvest,
            ) = originals

        self.assertEqual(len(harvest_calls), 1)
        self.assertAlmostEqual(harvest_calls[0]["z"], 10.0)


if __name__ == "__main__":
    unittest.main()
