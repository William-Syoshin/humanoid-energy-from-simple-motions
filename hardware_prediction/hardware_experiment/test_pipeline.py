"""Synthetic integration and adversarial tests; no physical serial ports."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import core
from protocol import preflight, audit_raw, freeze_design
from processing import prepare, power_grid, integrate, smooth_time_window
import analysis


def fixture(root, tid='T001', split='train', support='feet'):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    proof=root/'synthetic_evidence.txt'; proof.touch()
    cfg=core.configuration()
    cfg.update(mapping_verified=True,mixing_off_verified=True,max_initial_error_rad=.05,
               max_alignment_uncertainty_s=.02)
    for joint in cfg['joints']: joint.update(sign=1,min_rad=-1,max_rad=1,max_speed_rad_s=2)
    ev=dict(verified=True,evidence_path=str(proof),evidence_sha256=core.digest(proof))
    cfg['power_calibration'].update(ev,voltage_gain=1,voltage_offset=0,current_gain=1,current_offset=0)
    cfg['timing'].update(ev,receive_latency_s=.01,uncertainty_s=.005,max_receive_jitter_s=.01)
    cfg['throughput'].update(ev,validated_analysis_hz=10)
    recipe=dict(trial_id=tid,split=split,recipe_group='synthetic',approved=True,kind='hold',
        duration_s=1,pre_s=.2,post_s=.2,settle_s=0,support=support,posture='synthetic',
        center_rad=[0]*17,excitation=[],frames=[],success_gate='synthetic test only')
    raw=root/'raw';raw.mkdir()
    core.save(raw/'config.json',cfg);core.save(raw/'recipe.json',recipe)
    core.save(raw/'design.json',dict(config_sha256=core.digest(raw/'config.json'),recipes={tid:core.digest(raw/'recipe.json')}))
    core.save(raw/'meta.json',dict(status='complete',purpose='primary',execute=True,score_start=10.2,score_end=11.2,
                                 power_firmware='KHR3HV_POWER_V2',power_timestamp_source='fresh_update_observed_ms'))
    core.csv_save(raw/'power.csv',[dict(device_s=float(t),pc_s=float(t+10.01),voltage=12,current=2)
                                 for t in np.arange(0,1.51,.1)])
    core.csv_save(raw/'commands.csv',[dict(pc_s=10,response_s=10.01,ok=1,**{f'cmd_{i}':0 for i in range(17)})])
    core.csv_save(raw/'ad.csv',[dict(pc_s=float(t+10),request_s=float(t+10-.002),response_s=float(t+10+.002),ok=1,
                         ad1=0,ad2=0,ad3=300,ad4=330) for t in np.arange(0,1.51,.1065)])
    return cfg,recipe,raw


class PipelineTests(unittest.TestCase):
    def test_velocity_smoothing_uses_time_window(self):
        x5=np.arange(25,dtype=float)[:,None]
        x10=np.arange(50,dtype=float)[:,None]
        _,n5=smooth_time_window(x5,5,.4)
        _,n10=smooth_time_window(x10,10,.4)
        self.assertEqual(n5,3)
        self.assertEqual(n10,5)

    def test_capture_to_processing_with_fake_hardware(self):
        import sys,time,types
        import acquire
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c,r,raw=fixture(root/'fixture')
            session=root/'session';session.mkdir();(session/'recipes').mkdir()
            c['timing'].update(receive_latency_s=0,uncertainty_s=.005,max_receive_jitter_s=.02)
            r.update(duration_s=.7,pre_s=.15,post_s=.3)
            core.save(session/'config.json',c);core.save(session/'recipes'/'T001.json',r)
            core.csv_save(session/'trials.csv',[dict(trial_id='T001',recipe='recipes/T001.json',split='train',exclusion_reason='pending')])
            freeze_design(session,session/'design.json')
            class FakeRCB:
                RamAddr=SimpleNamespace(ConfigRamAddress=SimpleNamespace(value=0))
                @staticmethod
                def ServoData(i,s,p):return (i,s,p)
                def open(self,*a):return True
                def close(self):pass
                def getSinglePos(self,*a):return True,7500
                def getConfig(self):return 1
                def moveComToRamCmdSynchronize(self,*a):return True
                def setServoPos(self,*a):return True
                def adDataAddr(self,*a):return 0
                def moveRamToComCmdSynchronize(self,*a):return True,[0,0,0,0,44,1,74,1]
            real_sleep=time.sleep
            class FakeSerial:
                def __init__(self,*a,**kw):
                    self.zero=time.perf_counter();self.headers=iter([b'# FIRMWARE KHR3HV_POWER_V2\n',b'# TIMESTAMP fresh_update_observed_ms\n'])
                def readline(self):
                    header=next(self.headers,None)
                    if header is not None:return header
                    real_sleep(.02)
                    return f'{(time.perf_counter()-self.zero)*1000},12,2\n'.encode()
                def write(self,*a):pass
                def flush(self):pass
                def close(self):pass
            modules={'serial':types.SimpleNamespace(Serial=FakeSerial),
                     'Rcb4BaseLib':types.SimpleNamespace(Rcb4BaseLib=FakeRCB)}
            with patch.dict(sys.modules,modules),patch.object(acquire.time,'sleep',side_effect=lambda n:real_sleep(n) if n<1 else None):
                acquire.record(session/'config.json',session/'recipes'/'T001.json',root/'capture',True,'primary',session/'design.json')
            qc=core.read(root/'capture'/'processed'/'qc.json')
            self.assertTrue(qc['usable']);self.assertAlmostEqual(qc['energy_J'],24*.7,places=5)
            cmd=core.rows(root/'capture'/'commands.csv');self.assertEqual(len(cmd),8)

    def test_legacy_sensor_only_rejected_before_serial_import(self):
        from acquire import record
        with tempfile.TemporaryDirectory() as d:
            c,r,raw=fixture(d)
            with self.assertRaisesRegex(ValueError,'acknowledged commands'):
                record(raw/'config.json',raw/'recipe.json',Path(d)/'capture',False)
            self.assertFalse((Path(d)/'capture').exists())

    def test_preflight_collects_missing_items(self):
        c=core.configuration()
        from design import planned_recipes
        result=preflight(c,planned_recipes()[0],True)
        self.assertFalse(result['ready']); self.assertGreater(len(result['errors']),5)

    def test_mismatched_rates_known_energy_and_schema(self):
        with tempfile.TemporaryDirectory() as d:
            c,r,raw=fixture(d);out=Path(d)/'processed'
            prepare(raw,raw/'config.json',out)
            qc=core.read(out/'qc.json')
            self.assertAlmostEqual(qc['energy_J'],24,places=8)
            self.assertLess(qc['energy_difference_fraction'],1e-10)
            data=core.rows(out/'aligned.csv')
            self.assertNotIn('tilt_rad',data[0]);self.assertNotIn('q_0',data[0])
            self.assertIn('cmd_16',data[0]);self.assertEqual(data[0]['support'],'feet')

    def test_area_preservation_variable_power(self):
        t=np.arange(0,2.001,.013);p=12+2*np.sin(t*12)
        target=np.linspace(.1,1.9,19);pg=power_grid(t,p,target,.1)
        clipped=np.r_[.1,t[(t>.1)&(t<1.9)],1.9]
        expected=integrate(np.interp(clipped,t,p),clipped)
        self.assertAlmostEqual(integrate(pg,target),expected,places=10)

    def test_signed_power_not_rectified(self):
        t=np.array([0.,.1,.2,.3]);p=np.array([2.,-1.,-1.,2.])
        result=power_grid(t,p,t,.2)
        self.assertTrue(np.any(result<0))
        self.assertAlmostEqual(integrate(result,t),integrate(p,t))

    def test_unvalidated_tilt_never_silently_added(self):
        with tempfile.TemporaryDirectory() as d:
            c,r,raw=fixture(d);c['analysis_schema']['tilt']='fusion'
            self.assertFalse(preflight(c,r,True)['ready'])

    def test_power_gap_rejected(self):
        with self.assertRaisesRegex(ValueError,'Dropout'):
            power_grid(np.array([0.,.1,1.]),np.ones(3),np.linspace(0,1,11),.25)

    def test_reset_clock_and_receive_jitter_rejected(self):
        for mode in ('reset','jitter'):
            with tempfile.TemporaryDirectory() as d:
                c,r,raw=fixture(d);samples=core.rows(raw/'power.csv')
                samples[5]['device_s']=0 if mode=='reset' else samples[5]['device_s']
                if mode=='jitter':samples[5]['pc_s']=float(samples[5]['pc_s'])+.04
                (raw/'power.csv').unlink();core.csv_save(raw/'power.csv',samples)
                with self.assertRaises(ValueError):prepare(raw,raw/'config.json',Path(d)/'out')

    def test_design_has_coverage_and_separate_validation(self):
        from design import planned_recipes
        recipes=planned_recipes()
        self.assertEqual([sum(r['split']==s for r in recipes) for s in ('train','validation','test')],[120,24,21])
        for joint in core.NAMES:
            self.assertEqual(sum(r['split']=='train' and r['level']=='L1' and r['excitation'][0]['joint']==joint for r in recipes),3)
        self.assertEqual(sum(r['level']=='L2' and r['split']=='train' for r in recipes),30)
        self.assertTrue(all(not r['approved'] for r in recipes))

    def test_timing_calibration_preserves_unverified_state(self):
        from calibration import timing_calibration
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'points.csv';o=Path(d)/'result.json'
            core.csv_save(p,[dict(device_s=t,pc_reference_s=t+10,pc_receive_s=t+10.02) for t in (0,1,2)])
            timing_calibration(p,o,.001);result=core.read(o)
            self.assertFalse(result['verified']);self.assertAlmostEqual(result['receive_latency_s'],.02)

    def test_unknown_support_and_missing_clock_block(self):
        with tempfile.TemporaryDirectory() as d:
            c,r,raw=fixture(d,support='back')
            self.assertFalse(preflight(c,r,True)['ready'])
            r['support']='ground';c['timing']['verified']=False
            self.assertFalse(preflight(c,r,True)['ready'])

    def test_registration_hash_tampering_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            c,r,raw=fixture(d);out=Path(d)/'processed';prepare(raw,raw/'config.json',out)
            # Simulated corruption is test-owned data only.
            with (out/'aligned.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'changed aligned'):
                core.register(d,raw,out,1,'',Path(d)/'manifest.csv')

    def test_fit_does_not_read_test_and_evaluate_checks_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);aligned=root/'aligned';aligned.mkdir();manifest=[]
            for tid,split in [('T001','train'),('T002','validation'),('T003','test')]:
                c,r,raw=fixture(root/tid,tid,split);out=root/tid/'processed';prepare(raw,raw/'config.json',out)
                import shutil
                shutil.copyfile(out/'aligned.csv',aligned/(tid+'.csv'))
                shutil.copyfile(out/'qc.json',aligned/(tid+'.qc.json'))
                manifest.append(dict(trial_id=tid,file=tid+'.csv',split=split,task='synthetic',level='L0',attempted=1,usable=1,success=1))
            core.csv_save(root/'trials.csv',manifest)
            args=SimpleNamespace(manifest=str(root/'trials.csv'),data=str(aligned),out=str(root/'models'))
            original=analysis.read_trial
            def guarded(meta,*a,**kw):
                if meta['split']=='test':raise AssertionError('Test leakage during fit')
                return original(meta,*a,**kw)
            with patch.object(analysis,'read_trial',side_effect=guarded):analysis.fit(args)
            args.models=args.out;args.out=str(root/'results');analysis.evaluate(args)
            summary=core.read(root/'results'/'summary.json')
            self.assertEqual(summary['valid_test_trials'],1)
            self.assertLess(summary['models']['R0']['mean_error_pct'],1e-8)
            manifest[0]['split']='test'
            changed=root/'changed.csv';core.csv_save(changed,manifest)
            args.manifest=str(changed);args.out=str(root/'invalid_results')
            with self.assertRaisesRegex(ValueError,'splits or tasks changed'):analysis.evaluate(args)

    def test_feedback_branch_missing_data_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            c,r,raw=fixture(d)
            c['feedback']=True;c['analysis_schema']['feedback']=True;c['throughput']['feedback']=True
            (raw/'config.json').unlink();core.save(raw/'config.json',c)
            (raw/'design.json').unlink();core.save(raw/'design.json',dict(config_sha256=core.digest(raw/'config.json'),recipes={'T001':core.digest(raw/'recipe.json')}))
            with self.assertRaises((ValueError,FileNotFoundError)):
                prepare(raw,raw/'config.json',Path(d)/'out')


if __name__=='__main__':unittest.main()
