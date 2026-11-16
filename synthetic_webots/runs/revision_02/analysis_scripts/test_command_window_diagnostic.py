"""Hand-calculable window/integration checks; no real model scores are read."""
import unittest
import numpy as np
from command_window_diagnostic import command_window, truncate_series, power_metrics


class CommandWindowTests(unittest.TestCase):
    def test_change_is_assigned_to_latter_sample_plus_one_second(self):
        t = np.arange(7, dtype=float)
        q = np.zeros((7, 17)); q[2:, 4] = .1
        w = command_window(t, q)
        self.assertEqual(w["last_command_change_s"], 2.0)
        self.assertEqual(w["end_s"], 3.0)
        self.assertEqual(w["duration_s"], 3.0)

    def test_no_change_retains_full_recording(self):
        t = np.array([.01, .21, 1.21, 4.01])
        w = command_window(t, np.zeros((4, 17)))
        self.assertIsNone(w["last_command_change_s"])
        self.assertEqual(w["duration_s"], 4.0)
        self.assertFalse(w["shortened"])

    def test_strict_threshold_and_any_joint(self):
        t = np.arange(6, dtype=float)
        q = np.zeros((6, 17)); q[1:, 0] = 1e-6
        self.assertIsNone(command_window(t, q)["last_command_change_s"])
        q[2:, 16] = 1.01e-6
        self.assertEqual(command_window(t, q)["last_command_change_s"], 2.0)

    def test_cap_at_last_recorded_time(self):
        t = np.array([.1, .5, 1.0])
        q = np.zeros((3, 17)); q[-1, 0] = .1
        self.assertEqual(command_window(t, q)["end_s"], 1.0)

    def test_full_series_boundary_interpolation(self):
        # Last command change is t=2, hence end=3 between recorded t=2 and 4.
        t = np.array([0., 2., 4., 6.])
        q = np.zeros((4, 17)); q[1:, 0] = .2
        w = command_window(t, q)
        tw, (truth, prediction) = truncate_series(t, [t.copy(), 2*t], w["end_s"])
        np.testing.assert_array_equal(tw, [0., 2., 3.])
        np.testing.assert_array_equal(truth, [0., 2., 3.])
        # Integrals of t and 2t over [0,3] are 4.5 and 9; relative error=100%.
        m = power_metrics(tw, truth, prediction)
        self.assertEqual(m["reference_energy_J"], 4.5)
        self.assertEqual(m["predicted_energy_J"], 9.0)
        self.assertEqual(m["absolute_relative_energy_error_pct"], 100.0)

    def test_constant_power_and_sample_pooled_squared_error(self):
        t = np.array([.01, 1.01, 2.01])
        m = power_metrics(t, np.full(3, 2.), np.full(3, 3.))
        self.assertAlmostEqual(m["reference_energy_J"], 4.)
        self.assertAlmostEqual(m["predicted_energy_J"], 6.)
        self.assertAlmostEqual(m["absolute_relative_energy_error_pct"], 50.)
        self.assertEqual(m["squared_power_error_sum"], 3.)
        self.assertEqual(m["power_RMSE_W"], 1.)

    def test_invalid_command_data_does_not_silently_fall_back(self):
        with self.assertRaises(ValueError):
            command_window(np.array([0., 1.]), np.zeros((2, 16)))
        with self.assertRaises(ValueError):
            command_window(np.array([0., 0.]), np.zeros((2, 17)))


if __name__ == "__main__":
    unittest.main()
