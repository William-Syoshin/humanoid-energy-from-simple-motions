"""Two-axis gravity-direction integration. Parameters must be measured.

No yaw estimate. Gravity sign must be initialized independently; two acceleration
channels cannot resolve upright versus inverted on their own.
"""
import math
import numpy as np


def estimate_accelerometer(ad_rows, calibration, max_gap):
    """Estimate upright-branch tilt from the calibrated AD3/AD4 accelerometer.

    The recovered 15-degree calibration establishes that AD3 decreases when the
    body tilts forward and AD4 increases when it tilts right.  This estimator is
    intentionally independent of AD1/AD2 gyro gain.  It is suitable for static
    and slow motion; raw gyro channels remain logged for later fusion studies.
    """
    required = ('ad3_zero', 'ad4_zero', 'ad3_per_g', 'ad4_per_g')
    if not calibration.get('verified') or any(calibration.get(k) is None for k in required):
        raise ValueError('Static accelerometer tilt calibration is incomplete')
    z3, z4, g3, g4 = (float(calibration[k]) for k in required)
    if not all(math.isfinite(v) for v in (z3, z4, g3, g4)) or g3 <= 0 or g4 <= 0:
        raise ValueError('Invalid static accelerometer tilt calibration')
    previous = None
    result = []
    for row in ad_rows:
        t = float(row['pc_s'])
        if not math.isfinite(t) or (previous is not None and (t <= previous or t-previous > max_gap)):
            raise ValueError('Orientation sample gap or clock error')
        forward_g = (z3-float(row['ad3']))/g3
        right_g = (float(row['ad4'])-z4)/g4
        horizontal = math.hypot(forward_g, right_g)
        clipped = min(horizontal, 1.0)
        vertical_g = math.sqrt(max(0.0, 1.0-clipped*clipped))
        result.append(dict(
            pc_s=t,
            tilt_rad=math.asin(clipped),
            pitch_rad=math.atan2(forward_g, math.hypot(right_g, vertical_g)),
            roll_rad=math.atan2(right_g, vertical_g),
            accel_forward_g=forward_g,
            accel_right_g=right_g,
            accel_horizontal_norm_g=horizontal,
            accel_valid=int(horizontal <= 1.0),
        ))
        previous = t
    return result

def estimate(ad_rows, calibration, max_gap):
    required=('accel_zero','accel_per_g','gyro_zero','gyro_rad_s_per_count',
              'initial_gravity','correction_rate_s','accel_norm_tolerance')
    if not calibration.get('verified') or any(calibration.get(k) is None for k in required):
        raise ValueError('Orientation calibration is incomplete')
    zeros=np.asarray(calibration['accel_zero'],dtype=float)
    gains=np.asarray(calibration['accel_per_g'],dtype=float)
    gz=np.asarray(calibration['gyro_zero'],dtype=float)
    gg=np.asarray(calibration['gyro_rad_s_per_count'],dtype=float)
    # Matrices map sensor channels into body x/y. This includes sensor polarity.
    am=np.asarray(calibration.get('accel_body_xy_matrix'),dtype=float)
    gm=np.asarray(calibration.get('gyro_body_xy_matrix'),dtype=float)
    g=np.asarray(calibration['initial_gravity'],dtype=float)
    if any(a.shape!=(2,) for a in (zeros,gains,gz,gg)) or am.shape!=(2,2) or gm.shape!=(2,2) or g.shape!=(3,):
        raise ValueError('Invalid orientation dimensions')
    if not all(np.isfinite(a).all() for a in (zeros,gains,gz,gg,am,gm,g)) or np.any(gains==0) or np.any(gg==0) or np.linalg.norm(g)<1e-9:
        raise ValueError('Invalid orientation calibration')
    rate=calibration['correction_rate_s']; tolerance=calibration['accel_norm_tolerance']
    if not math.isfinite(rate) or rate<0 or not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('Invalid correction parameters')
    g=g/np.linalg.norm(g); previous=None; result=[]
    for row in ad_rows:
        t=float(row['pc_s'])
        gyro=np.array([int(row['ad1']),int(row['ad2'])])
        gyro=np.where(gyro>=32768,gyro-65536,gyro)
        omega=np.r_[gm@((gyro-gz)*gg),0.0]
        dt=0 if previous is None else t-previous
        if not math.isfinite(t) or dt<0 or (previous is not None and dt==0) or dt>max_gap:
            raise ValueError('Orientation sample gap or clock error')
        # Rotate gravity into the new body frame using Rodrigues' formula.
        angle=np.linalg.norm(omega)*dt
        if angle:
            axis=-omega/np.linalg.norm(omega)
            g=g*math.cos(angle)+np.cross(axis,g)*math.sin(angle)+axis*np.dot(axis,g)*(1-math.cos(angle))
        xy=am@((np.array([float(row['ad3']),float(row['ad4'])])-zeros)/gains)
        norm2=float(xy@xy)
        corrected=False
        if np.isfinite(xy).all() and norm2<=1+tolerance:
            if norm2>1: xy/=math.sqrt(norm2); norm2=1
            measured=np.r_[xy,math.copysign(math.sqrt(max(0,1-norm2)),g[2])]
            weight=1-math.exp(-rate*dt)
            g=(1-weight)*g+weight*measured
            if np.linalg.norm(g)<1e-9: raise ValueError('Ambiguous gravity estimate')
            g/=np.linalg.norm(g); corrected=True
        result.append(dict(pc_s=t,tilt_rad=math.acos(float(np.clip(g[2],-1,1))),
            pitch_rad=math.atan2(-g[0],math.hypot(g[1],g[2])),
            roll_rad=math.atan2(g[1],g[2]),accel_correction=int(corrected)))
        previous=t
    return result
