import sys
import types
import unittest
from unittest import mock

import numpy as np


try:
    import ultralytics  # noqa: F401
except ImportError:
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=object)

import main
from arm_controller import CUT_GAP_M, FiveDOFArm, compute_cut_point, search_home_angles
from servo_driver import (
    DEFAULT_BAUD, DEFAULT_SERVO_BACKEND, DEFAULT_UART_PORT,
    GRIPPER_OPEN_PULSE, ServoDriver,
)
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
        self.ee_pos = np.array([0.25, 0.0, 0.18], dtype=float)
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

    def end_effector_pos(self):
        return self.ee_pos.copy()

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

    def test_reference_snapshot_ranks_three_closest_targets(self):
        arm = FiveDOFArm()
        arm.set_joint_angles(search_home_angles())
        samples = [
            detection(0.0, 0.0, 40.0, confidence=0.7),
            detection(0.5, 0.0, 40.5, confidence=0.8),
            detection(2.0, 0.0, 25.0, confidence=0.9),
            detection(2.5, 0.0, 24.5, confidence=0.8),
            detection(-2.0, 0.0, 30.0, confidence=0.85),
            detection(-2.5, 0.0, 30.5, confidence=0.8),
            detection(6.0, 0.0, 18.0, confidence=0.75),
            detection(6.5, 0.0, 18.5, confidence=0.8),
        ]
        for i, sample in enumerate(samples):
            sample["bbox_px"] = [10 + i, 20, 50 + i, 60]
            sample["center_px"] = [30 + i, 40]

        snapshot = main._build_reference_snapshot(samples, arm, captured_at=123.0)

        self.assertEqual(snapshot["captured_at"], 123.0)
        self.assertEqual(len(snapshot["targets"]), 3)
        self.assertEqual([target["rank"] for target in snapshot["targets"]], [1, 2, 3])
        self.assertEqual([target["distance_cm"] for target in snapshot["targets"]],
                         [18.25, 24.75, 30.25])
        np.testing.assert_allclose(snapshot["initial_joint_angles"],
                                   search_home_angles())

    def test_guided_harvest_cuts_one_centimetre_above_live_center(self):
        class FakeDetector:
            def detect(self, _frame):
                det = detection(0.0, 0.0, 10.0, confidence=0.95)
                det["bbox_px"] = [20, 20, 220, 220]
                det["center_px"] = [120, 120]
                return [det]

            def annotate(self, frame, _detections):
                return frame.copy()

        class FakeCapture:
            def read(self):
                return True, np.zeros((240, 240, 3), dtype=np.uint8)

        arm = FakeArm(reachable=True, ik_result=(True, 0.0, 1))
        driver = FakeDriver()
        reference = detection(0.0, 0.0, 10.0, confidence=0.95)
        reference.update({
            "rank": 1,
            "bbox_px": [20, 20, 220, 220],
            "center_px": [120, 120],
        })

        with mock.patch.object(main.cv2, "imshow"), \
                mock.patch.object(main.cv2, "waitKey", return_value=-1):
            harvest_ok, reason = main._guided_harvest_target(
                arm, driver, FakeDetector(), FakeCapture(), reference)

        expected_cut = arm.ee_pos + np.array([0.0, 0.0, main.GUIDED_CUT_OFFSET_M])
        self.assertTrue(harvest_ok, reason)
        np.testing.assert_allclose(arm.ik_target, expected_cut)
        self.assertIn(("close", 500), driver.calls)

    def test_visual_servo_moves_relative_to_image_error(self):
        arm = FakeArm()
        det = detection(0.0, 0.0, 30.0, confidence=0.9)
        det["bbox_px"] = [250, 150, 330, 230]
        det["center_px"] = [480, 360]  # right and below image centre

        target = main._visual_servo_target_point(arm, det, (480, 640, 3))
        delta = target - arm.end_effector_pos()

        self.assertGreater(delta[0], 0.0)   # approach forward while tomato is small
        self.assertLess(delta[1], 0.0)      # image right maps to arm -Y
        self.assertLess(delta[2], 0.0)      # image down maps to arm -Z

    def test_servo_self_test_commands_gripper_and_base(self):
        driver = FakeDriver()

        main._run_servo_self_test(driver)

        move_calls = [call for call in driver.calls if call[0] == "move"]
        self.assertGreaterEqual(len(move_calls), 3)
        self.assertIn(("open", main.SERVO_SELF_TEST_DURATION_MS), driver.calls)
        self.assertIn(("close", main.SERVO_SELF_TEST_DURATION_MS), driver.calls)
        self.assertTrue(any(call[1][6] != main.SEARCH_HOME_PULSES[6]
                            for call in move_calls))

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

    def test_harvest_approach_commands_all_five_servos(self):
        arm = FiveDOFArm()
        arm.set_joint_angles(search_home_angles())
        driver = FakeDriver()
        locked = {"x": 0.0, "y": 0.0, "z": 10.0}

        harvest_ok, reason = main._execute_harvest(arm, driver, locked, frame=None)

        approach_moves = [
            call for call in driver.calls
            if call[0] == "move" and call[2] == main.MOVE_DURATION_MS // 25
        ]
        self.assertTrue(harvest_ok, reason)
        self.assertEqual(reason, "cut complete")
        self.assertEqual(len(approach_moves), 26)
        self.assertTrue(all(set(call[1]) == {1, 3, 4, 5, 6}
                            for call in approach_moves))
        self.assertTrue(all(call[1][1] == GRIPPER_OPEN_PULSE
                            for call in approach_moves))
        self.assertIn(("close", 500), driver.calls)

    def test_harvest_with_sdk_backend_sends_five_hardware_servo_commands(self):
        board = types.ModuleType("HiwonderSDK.Board")
        board.setBusServoPulse = mock.Mock()
        package = types.ModuleType("HiwonderSDK")

        with mock.patch.dict(sys.modules, {
            "HiwonderSDK": package,
            "HiwonderSDK.Board": board,
        }):
            arm = FiveDOFArm()
            arm.set_joint_angles(search_home_angles())
            driver = ServoDriver(mode="real", backend="sdk")

            harvest_ok, reason = main._execute_harvest(
                arm, driver, {"x": 0.0, "y": 0.0, "z": 10.0}, frame=None)

        approach_calls = [
            call.args for call in board.setBusServoPulse.call_args_list
            if call.args[2] == main.MOVE_DURATION_MS // 25
        ]
        self.assertTrue(harvest_ok, reason)
        self.assertEqual(reason, "cut complete")
        self.assertEqual(driver.backend, "sdk")
        self.assertEqual(len(approach_calls), 26 * 5)
        self.assertEqual({args[0] for args in approach_calls[:5]}, {1, 3, 4, 5, 6})
        self.assertTrue(all(args[1] == GRIPPER_OPEN_PULSE
                            for args in approach_calls if args[0] == 1))
        board.setBusServoPulse.assert_any_call(1, GRIPPER_OPEN_PULSE, 400)
        board.setBusServoPulse.assert_any_call(1, 700, 500)

    def test_uart_backend_writes_one_packet_per_servo(self):
        writes = []

        class FakeSerial:
            def __init__(self, port, baud, timeout, parity, stopbits):
                self.port = port
                self.baud = baud
                self.timeout = timeout
                self.parity = parity
                self.stopbits = stopbits

            def write(self, packet):
                writes.append(packet)

            def close(self):
                pass

        fake_serial = types.SimpleNamespace(
            Serial=FakeSerial,
            PARITY_NONE="N",
            STOPBITS_ONE=1,
        )

        with mock.patch.dict(sys.modules, {"serial": fake_serial}):
            driver = ServoDriver(mode="real", backend="uart",
                                 uart_port="/dev/fake", baud=12345)
            driver.move_servos({6: 500, 5: 600, 4: 700, 3: 400, 1: 200},
                               duration_ms=321)

        self.assertEqual(driver.backend, "uart")
        self.assertEqual(len(writes), 5)
        self.assertTrue(all(packet.startswith(b"\x55\x55") for packet in writes))
        self.assertEqual([packet[2] for packet in writes], [6, 5, 4, 3, 1])

    def test_auto_backend_falls_back_to_uart_when_sdk_is_unavailable(self):
        writes = []

        class FakeSerial:
            def __init__(self, *_args, **_kwargs):
                pass

            def write(self, packet):
                writes.append(packet)

            def close(self):
                pass

        fake_serial = types.SimpleNamespace(
            Serial=FakeSerial,
            PARITY_NONE="N",
            STOPBITS_ONE=1,
        )

        with mock.patch("servo_driver.importlib.import_module",
                        side_effect=ImportError("no sdk")), \
                mock.patch.dict(sys.modules, {"serial": fake_serial}):
            driver = ServoDriver(mode="real", backend="auto")
            driver.move_servo(6, 500, duration_ms=123)

        self.assertEqual(driver.backend, "uart")
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][2], 6)

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

            def set(self, *_args):
                return True

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
        snapshot = {
            "captured_at": 0.0,
            "initial_joint_angles": np.zeros(5),
            "initial_pulses": {},
            "targets": [{
                "rank": 1,
                "observations": 3,
                "confidence": 0.9,
                "bbox_px": [100, 100, 180, 180],
                "center_px": [140, 140],
                "xyz_cm": {"x": 0.0, "y": 0.0, "z": 10.0},
                "distance_cm": 10.0,
            }],
        }
        originals = (
            main.TomatoDetector,
            main.cv2.VideoCapture,
            main.cv2.imshow,
            main.cv2.waitKey,
            main.cv2.destroyAllWindows,
            main.time.time,
            main.time.sleep,
            main._collect_validation_snapshot,
            main._guided_harvest_target,
        )

        def fake_wait_key(_delay):
            wait_calls["count"] += 1
            if harvest_calls:
                return ord("q")
            return ord("q") if wait_calls["count"] >= 35 else -1

        def fake_collect(_cap, _detector, _arm, validation_seconds,
                         min_observations):
            self.assertEqual(validation_seconds, main.CONFIRM_SECONDS)
            self.assertEqual(min_observations, main.CONFIRM_FRAMES)
            return snapshot, np.zeros((480, 640, 3), dtype=np.uint8)

        def fake_guided(_arm, _driver, _detector, _cap, target):
            harvest_calls.append(dict(target))
            return True, "test harvest"

        try:
            main.TomatoDetector = FakeDetector
            main.cv2.VideoCapture = FakeCapture
            main.cv2.imshow = lambda *_args, **_kwargs: None
            main.cv2.waitKey = fake_wait_key
            main.cv2.destroyAllWindows = lambda: None
            main.time.time = clock.time
            main.time.sleep = clock.sleep
            main._collect_validation_snapshot = fake_collect
            main._guided_harvest_target = fake_guided

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
                main._collect_validation_snapshot,
                main._guided_harvest_target,
            ) = originals

        self.assertEqual(len(harvest_calls), 1)
        self.assertEqual(harvest_calls[0]["rank"], 1)

    def test_main_defaults_to_hardware_mode(self):
        with mock.patch.object(sys, "argv", ["main.py"]), \
                mock.patch.object(main, "run_harvesting") as run_harvesting:
            main.main()

        run_harvesting.assert_called_once_with(
            sim_mode=False,
            confirm_seconds=main.CONFIRM_SECONDS,
            confirm_frames=main.CONFIRM_FRAMES,
            camera_index=main.CAMERA_INDEX,
            camera_source=main.DEFAULT_CAMERA_SOURCE,
            ros_image_topic=main.DEFAULT_ROS_IMAGE_TOPIC,
            ros_frame_timeout=main.ROS_FRAME_TIMEOUT_S,
            servo_backend=DEFAULT_SERVO_BACKEND,
            uart_port=DEFAULT_UART_PORT,
            baud=DEFAULT_BAUD,
            servo_self_test=False)

    def test_main_passes_servo_backend_cli_options(self):
        with mock.patch.object(sys, "argv", [
                "main.py", "--camera-source", "ros", "--ros-image-topic",
                "/usb_cam/image_raw", "--ros-frame-timeout", "2.5",
                "--servo-backend", "sdk", "--uart-port", "/dev/test",
                "--baud", "57600"]), \
                mock.patch.object(main, "run_harvesting") as run_harvesting:
            main.main()

        run_harvesting.assert_called_once_with(
            sim_mode=False,
            confirm_seconds=main.CONFIRM_SECONDS,
            confirm_frames=main.CONFIRM_FRAMES,
            camera_index=main.CAMERA_INDEX,
            camera_source="ros",
            ros_image_topic="/usb_cam/image_raw",
            ros_frame_timeout=2.5,
            servo_backend="sdk",
            uart_port="/dev/test",
            baud=57600,
            servo_self_test=False)

    def test_ros_camera_source_reads_image_topic_frame(self):
        frame = np.ones((3, 4, 3), dtype=np.uint8)

        class FakeCore:
            @staticmethod
            def is_initialized():
                return False

        class FakeSubscriber:
            def __init__(self, topic, msg_type, callback, **_kwargs):
                self.topic = topic
                self.msg_type = msg_type
                callback(object())

            def unregister(self):
                pass

        fake_rospy = types.SimpleNamespace(
            core=FakeCore,
            init_node=mock.Mock(),
            Subscriber=FakeSubscriber,
            is_shutdown=lambda: False,
            sleep=lambda _seconds: None,
        )

        class FakeBridge:
            def imgmsg_to_cv2(self, _msg, desired_encoding="bgr8"):
                self.encoding = desired_encoding
                return frame

        fake_cv_bridge = types.SimpleNamespace(CvBridge=FakeBridge)
        fake_sensor_msgs = types.ModuleType("sensor_msgs")
        fake_sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
        fake_sensor_msgs_msg.Image = type("Image", (), {})

        with mock.patch.dict(sys.modules, {
            "rospy": fake_rospy,
            "cv_bridge": fake_cv_bridge,
            "sensor_msgs": fake_sensor_msgs,
            "sensor_msgs.msg": fake_sensor_msgs_msg,
        }):
            source, selected = main._open_ros_camera("/usb_cam/image_raw", 0.1)

        self.assertEqual(selected, "/usb_cam/image_raw")
        self.assertTrue(source.isOpened())
        ok, got = source.read()
        self.assertTrue(ok)
        np.testing.assert_array_equal(got, frame)
        fake_rospy.init_node.assert_called_once()

    def test_camera_auto_open_tries_detected_devices_then_fallbacks(self):
        opened = []
        released = []

        class FakeCapture:
            def __init__(self, idx, backend):
                self.idx = idx
                self.backend = backend
                opened.append((idx, backend))

            def set(self, *_args):
                return True

            def isOpened(self):
                return self.idx == 2

            def read(self):
                return True, np.zeros((2, 2, 3), dtype=np.uint8)

            def release(self):
                released.append(self.idx)

        with mock.patch.object(main.os, "listdir", return_value=["video2"]), \
                mock.patch.object(main.cv2, "VideoCapture", FakeCapture):
            cap, idx = main._open_camera(-1)

        self.assertEqual(idx, 2)
        self.assertEqual(cap.idx, 2)
        self.assertEqual(opened[0], (2, main.cv2.CAP_V4L2))
        self.assertNotIn(2, released)

    def test_camera_auto_open_skips_devices_that_do_not_stream_frames(self):
        opened = []
        released = []

        class FakeCapture:
            def __init__(self, idx, backend):
                self.idx = idx
                self.backend = backend
                opened.append(idx)

            def set(self, *_args):
                return True

            def isOpened(self):
                return True

            def read(self):
                if self.idx == 25:
                    return True, np.zeros((2, 2, 3), dtype=np.uint8)
                return False, None

            def release(self):
                released.append(self.idx)

        with mock.patch.object(main.os, "listdir",
                               return_value=["video24", "video25"]), \
                mock.patch.object(main.cv2, "VideoCapture", FakeCapture):
            cap, idx = main._open_camera(-1)

        self.assertEqual(idx, 25)
        self.assertEqual(cap.idx, 25)
        self.assertEqual(opened[:2], [24, 25])
        self.assertIn(24, released)
        self.assertNotIn(25, released)


if __name__ == "__main__":
    unittest.main()
