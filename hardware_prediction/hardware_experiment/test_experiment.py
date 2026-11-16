import tempfile
import unittest
from pathlib import Path
import numpy as np
from core import configuration, initialize, read, compile_recipe, interpolate, save, csv_save, prepare

class ExperimentTests(unittest.TestCase):
    def configured(self):
        c=configuration(); c.update(mapping_verified=True,mixing_off_verified=True)
        for j in c['joints']: j.update(sign=1,min_rad=-1,max_rad=1,max_speed_rad_s=2)
        r=dict(approved=True,kind='sine',duration_s=8,pre_s=2,post_s=2,settle_s=0,
               support='fixture',center_rad=[0]*17,
               excitation=[dict(joint='left_knee_pitch',amplitude_rad=.1,frequency_Hz=.25,phase_rad=0)])
        return c,r
    def test_init_counts_and_no_overwrite(self):
        from core import rows
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'session'; initialize(p)
            rr=rows(p/'trials.csv')
            self.assertEqual([sum(r['split']==s for r in rr) for s in ('train','validation','test')],[120,24,21])
            with self.assertRaises(FileExistsError): initialize(p)
    def test_envelope_and_limits(self):
        c,r=self.configured(); data=compile_recipe(c,r)
        self.assertAlmostEqual(data[0]['cmd_2'],0)
        self.assertAlmostEqual(data[-1]['cmd_2'],0)
        self.assertGreater(max(abs(row['cmd_2']) for row in data),.05)
        r['excitation'][0]['amplitude_rad']=2
        with self.assertRaises(ValueError): compile_recipe(c,r)
    def test_missing_calibration(self):
        c,r=self.configured(); c['joints'][0]['sign']=None
        with self.assertRaises(ValueError): compile_recipe(c,r)
    def test_no_extrapolation_or_gap_bridge(self):
        with self.assertRaises(ValueError): interpolate([0,1],[0,1],[-1,.5],2)
        with self.assertRaises(ValueError): interpolate([0,1],[0,1],[.5],.1)
    def test_gravity_fusion_and_missing_parameters(self):
        from orientation import estimate
        cal=dict(verified=True,accel_zero=[0,0],accel_per_g=[1,1],gyro_zero=[0,0],
            gyro_rad_s_per_count=[.01,.01],initial_gravity=[0,0,1],
            accel_body_xy_matrix=[[1,0],[0,1]],gyro_body_xy_matrix=[[1,0],[0,1]],
            correction_rate_s=0,accel_norm_tolerance=0)
        data=[dict(pc_s=k*.1,ad1=100,ad2=0,ad3=0,ad4=0) for k in range(11)]
        estimate_rows=estimate(data,cal,.2)
        self.assertAlmostEqual(estimate_rows[-1]['tilt_rad'],1.0,places=6)
        cal['gyro_zero']=None
        with self.assertRaises(ValueError): estimate(data,cal,.2)
    def test_accelerometer_only_tilt(self):
        from orientation import estimate_accelerometer
        cal=dict(verified=True,ad3_zero=306.4778761061947,ad4_zero=336.98230088495575,
                 ad3_per_g=203.5,ad4_per_g=209.4)
        forward=[dict(pc_s=0.0,ad3=246.01769911504425,ad4=336.98230088495575)]
        row=estimate_accelerometer(forward,cal,.25)[0]
        self.assertGreater(row['pitch_rad'],0)
        self.assertAlmostEqual(np.degrees(row['tilt_rad']),17.2836,places=3)
        right=[dict(pc_s=0.0,ad3=306.4778761061947,ad4=390.070796460177)]
        row=estimate_accelerometer(right,cal,.25)[0]
        self.assertGreater(row['roll_rad'],0)
        self.assertAlmostEqual(np.degrees(row['tilt_rad']),14.6863,places=3)
    def test_legacy_prepare_requires_primary_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw=root/'raw'; raw.mkdir()
            c,_=self.configured()
            c['power_calibration'].update(verified=True,voltage_gain=1,voltage_offset=0,current_gain=1,current_offset=0)
            c['alignment'].update(scale=1,offset_s=10,residual_s=.001,method='synthetic test')
            save(root/'config.json',c)
            save(raw/'meta.json',dict(status='complete',score_start=10.1,score_end=10.9))
            save(raw/'recipe.json',dict(support='fixture'))
            csv_save(raw/'power.csv',[dict(device_s=float(t),voltage=12,current=2) for t in np.linspace(0,1,21)])
            csv_save(raw/'commands.csv',[dict(pc_s=10,ok=1,**{f'cmd_{i}':0 for i in range(17)})])
            with self.assertRaisesRegex(ValueError,'immutable acquisition config'):
                # Old raw exports without their saved config cannot be promoted.
                save(raw/'config.json',configuration())
                prepare(raw,root/'config.json',root/'processed')

if __name__=='__main__': unittest.main()
