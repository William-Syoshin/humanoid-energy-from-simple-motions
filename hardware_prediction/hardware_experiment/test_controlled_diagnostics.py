import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import controlled_diagnostics as d


class ControlledExperimentTests(unittest.TestCase):
    def test_trajectory_matching_and_return(self):
        cfg = d.read(d.SOURCE / 'config.json')
        template = d.read(d.SOURCE / 'recipe.json')
        for n in (1, 2):
            trajectories = {}
            for c in d.CONDITIONS:
                r = d.make_recipe(template, c, n)
                commands = d.compile_recipe(cfg, r)
                q = np.array([[x[f'cmd_{j}'] for j in range(17)] for x in commands])
                delta = q - r['center_rad']
                np.testing.assert_allclose(delta[[0, -1]], 0, atol=1e-14)
                self.assertEqual(len(q), 101)
                for j in range(17):
                    if j not in (10, 12, 13, 15):
                        np.testing.assert_allclose(delta[:, j], 0, atol=1e-14)
                    else:
                        self.assertGreaterEqual(delta[:, j].min(), 0)
                        self.assertAlmostEqual(delta[:, j].max(), np.deg2rad(10))
                        self.assertLess(abs(np.diff(q[:, j])/.1).max(), .14)
                trajectories[c] = delta
            for c in 'BCD':
                np.testing.assert_allclose(np.sort(trajectories['A'], axis=0),
                                           np.sort(trajectories[c], axis=0), atol=1e-14)
            for c in 'AB':
                delta = trajectories[c]
                overlap = (delta[:, 10] > 1e-10) & (delta[:, 13] > 1e-10)
                self.assertEqual(bool(overlap.any()), c == 'A')

    def test_plan_integrity_and_hardware_gate(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)/'session'
            schedule = d.plan(root, 2)
            self.assertEqual(len(schedule), 8)
            tid = schedule[0]['trial_id']
            with patch('acquire.record') as recorder:
                d.run(root, tid, None, False)
                recorder.assert_not_called()
                with self.assertRaises(ValueError):
                    d.run(root, tid, 'supine', True, True)
                recorder.assert_not_called()
            with self.assertRaises(FileExistsError):
                d.plan(root, 2)
            (root/'config.json').write_text('{}')
            with self.assertRaises(ValueError):
                d.verify(root)

    def test_empty_evaluation_is_not_a_result(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)/'session'
            d.plan(root, 2)
            d.evaluate(root, Path(t)/'result')
            report = d.read(Path(t)/'result/interpretation.json')
            self.assertFalse(report['complete'])
            self.assertEqual(report['processed_trials'], 0)

    def test_evaluation_on_synthetic_constant_power(self):
        # Synthetic data live only in a temporary directory, never the experiment.
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)/'session'
            schedule = d.plan(root, 2)
            for entry in schedule[:2]:
                tid = entry['trial_id']; raw = root/'raw'/tid
                raw.mkdir(parents=True)
                d.save(raw/'observation.json', {'success':True, 'note':'synthetic unit test'})
                aligned = root/'regression_data/aligned'
                aligned.mkdir(parents=True, exist_ok=True)
                recipe = d.read(root/entry['recipe'])
                data = d.compile_recipe(d.read(root/'config.json'), recipe)
                for row in data:
                    row.update(voltage_V=10, current_A=1, power_W=10, tilt_rad=0, support='feet')
                    row.update({f'q_{j}':row[f'cmd_{j}'] for j in range(17)})
                p = aligned/f'{tid}.csv'
                d.csv_save(p, data)
                d.save(p.with_suffix('.qc.json'), {'usable':True, 'aligned_sha256':d.digest(p)})
            out = Path(t)/'results'
            d.evaluate(root, out)
            metrics = d.rows(out/'per_trial_metrics.csv')
            self.assertEqual(len(metrics), 8)
            for metric in metrics:
                self.assertAlmostEqual(float(metric['actual_J']), 100)
            contrasts = d.rows(out/'within_repetition_contrasts.csv')
            self.assertEqual(len(contrasts), 1)
            self.assertEqual(contrasts[0]['contrast'], 'A-B')


if __name__ == '__main__':
    unittest.main()
