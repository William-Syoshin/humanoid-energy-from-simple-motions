"""Synthetic counterexamples for mechanical success checks (no Webots needed)."""
import json
import math
import unittest

import numpy as np

from motion_audit import audit_motion


def trajectory(action="turn_90", angle=math.pi / 2, initial_yaw=0.0):
    t = np.linspace(0, 4, 401)
    heading = initial_yaw + angle * np.minimum(t / 2.5, 1)
    rot = np.zeros((len(t), 3, 3))
    rot[:, 0, 0] = rot[:, 1, 1] = np.cos(heading)
    rot[:, 1, 0] = np.sin(heading); rot[:, 0, 1] = -np.sin(heading)
    rot[:, 2, 2] = 1
    position = np.column_stack((np.zeros(len(t)), np.zeros(len(t)), np.full(len(t), .34)))
    velocity = np.zeros((len(t), 6)); velocity[:, 5] = np.gradient(heading, t)
    contact = np.zeros((len(t), 7)); contact[:, 4:6] = 8
    return dict(t=t, h=np.zeros(len(t)), yaw=np.arctan2(rot[:, 1, 0], rot[:, 0, 0]),
        root_orientation=rot, root_velocity=velocity, root_omega=velocity[:, 3:].copy(),
        position=position, origin=position[0].copy(), contact=contact,
        object_position=np.zeros((len(t), 3)), object_origin=np.zeros(3),
        qcmd=np.zeros((len(t), 17)), spec_json=json.dumps(dict(action=action)))


def set_pitch(data, pitch):
    pitch = np.broadcast_to(pitch, data["t"].shape)
    rot = np.zeros((len(pitch), 3, 3))
    rot[:, 0, 0] = rot[:, 2, 2] = np.cos(pitch)
    rot[:, 0, 2] = np.sin(pitch); rot[:, 2, 0] = -np.sin(pitch)
    rot[:, 1, 1] = 1
    data["root_orientation"] = rot
    data["h"] = 1 - np.cos(pitch)
    data["yaw"] = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
    data["root_velocity"][:, 4] = np.gradient(pitch, data["t"])
    data["root_omega"] = data["root_velocity"][:, 3:].copy()


class MotionAuditTests(unittest.TestCase):
    def test_correct_turn_and_heading_wrap(self):
        for target, name in [(math.pi / 2, "turn_90"), (math.pi, "turn_180")]:
            data = trajectory(name, target, initial_yaw=math.radians(170))
            report = audit_motion(data)
            self.assertTrue(report["revised_motion_pass"])
            self.assertAlmostEqual(report["metrics"]["signed_unwrapped_yaw_change_rad"], target)

    def test_reverse_turn_legacy_passes_revised_fails(self):
        report = audit_motion(trajectory(angle=-math.pi / 2))
        self.assertTrue(report["legacy_endpoint_pass"])
        self.assertFalse(report["revised_motion_pass"])
        self.assertIn("turn_positive_direction", report["failed_revised_gates"])

    def test_extra_full_revolution_is_not_a_correct_turn(self):
        report = audit_motion(trajectory(angle=2 * math.pi + math.pi / 2))
        self.assertTrue(report["legacy_endpoint_pass"])
        self.assertFalse(report["revised_motion_pass"])
        self.assertIn("turn_target_error", report["failed_revised_gates"])

    def test_inverted_torso_does_not_count_as_upright(self):
        data = trajectory("walk", angle=0)
        data["position"][:, 0] = np.linspace(0, .1, len(data["t"]))
        set_pitch(data, math.pi)
        report = audit_motion(data)
        self.assertFalse(report["revised_motion_pass"])
        self.assertFalse(report["legacy_endpoint_pass"])

    def test_get_up_requires_sustained_standing(self):
        data = trajectory("get_up", angle=0)
        set_pitch(data, np.where(data["t"] < 4, math.pi / 2, 0))
        data["position"][:, 2] = np.where(data["t"] < 4, .1, .34)
        report = audit_motion(data)
        self.assertTrue(report["legacy_endpoint_pass"])
        self.assertFalse(report["revised_motion_pass"])
        self.assertIn("terminal_upright_height", report["failed_revised_gates"])

    def test_successful_get_up_and_initial_state_gate(self):
        data = trajectory("get_up", angle=0)
        factor = np.minimum(data["t"] / 2, 1)
        set_pitch(data, (1 - factor) * math.pi / 2)
        data["position"][:, 2] = .1 + .24 * factor
        data["root_velocity"][:, 2] = np.gradient(data["position"][:, 2], data["t"])
        self.assertTrue(audit_motion(data)["revised_motion_pass"])
        never_lay_down = trajectory("get_up", angle=0)
        self.assertFalse(audit_motion(never_lay_down)["revised_motion_pass"])

    def test_lie_down_requires_motion_from_upright_and_rest(self):
        data = trajectory("lie_down", angle=0)
        factor = np.minimum(data["t"] / 2, 1)
        set_pitch(data, factor * math.pi / 2)
        data["position"][:, 2] = .34 - .24 * factor
        data["root_velocity"][:, 2] = np.gradient(data["position"][:, 2], data["t"])
        self.assertTrue(audit_motion(data)["revised_motion_pass"])
        data["root_velocity"][-101:, 0] = .03
        self.assertFalse(audit_motion(data)["revised_motion_pass"])

    def test_stop_with_terminal_drift_fails(self):
        data = trajectory("stop", angle=0)
        data["position"][:, 0] = .05 * data["t"]
        data["root_velocity"][:, 0] = .05
        report = audit_motion(data)
        self.assertTrue(report["legacy_endpoint_pass"])
        self.assertFalse(report["revised_motion_pass"])
        self.assertIn("terminal_horizontal_speed", report["failed_revised_gates"])

    def test_walk_does_not_require_stopping(self):
        data = trajectory("walk", angle=0)
        data["position"][:, 0] = .05 * data["t"]
        data["root_velocity"][:, 0] = .05
        report = audit_motion(data)
        self.assertTrue(report["revised_motion_pass"])
        self.assertFalse(report["revised_gates"]["terminal_horizontal_speed"])

    def test_final_support_time_fraction(self):
        data = trajectory()
        data["contact"][data["t"] > 3.8, 4] = 0
        report = audit_motion(data)
        self.assertFalse(report["revised_motion_pass"])
        self.assertLess(report["metrics"]["terminal_both_feet_supported_time_fraction"], .95)

    def test_legacy_fallback_is_explicit(self):
        data = trajectory("stop", angle=0)
        data["position"][:, 0] = .1 * np.minimum(data["t"] / 2, 1)
        del data["root_velocity"]; del data["root_orientation"]; del data["qcmd"]
        report = audit_motion(data)
        self.assertTrue(report["revised_motion_pass"])
        self.assertIn("finite differences", report["signal_sources"]["velocity"])
        self.assertFalse(report["command_diagnostics"]["qcmd_present"])

    def test_nonfinite_or_inconsistent_orientation_rejected(self):
        data = trajectory(); data["position"][10, 0] = np.nan
        with self.assertRaises(ValueError): audit_motion(data)
        data = trajectory(); data["h"][10] = .9
        with self.assertRaises(ValueError): audit_motion(data)

    def test_insufficient_recording_cannot_satisfy_one_second_hold(self):
        data = trajectory()
        data["t"] = data["t"] / 10
        report = audit_motion(data)
        self.assertFalse(report["revised_motion_pass"])
        self.assertIn("full_terminal_window", report["failed_revised_gates"])

    def test_inclusive_tilt_boundary_and_action_conflict(self):
        data = trajectory("walk", angle=0)
        data["position"][:, 0] = .05 * data["t"]
        set_pitch(data, math.radians(20))
        self.assertTrue(audit_motion(data)["revised_motion_pass"])
        with self.assertRaises(ValueError): audit_motion(data, action="get_up")


if __name__ == "__main__":
    unittest.main()
