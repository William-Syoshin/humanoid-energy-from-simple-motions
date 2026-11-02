"""Independent timing calibration, never inferred from power/prediction agreement."""
import numpy as np
from core import rows, save, digest


def timing_calibration(points, output, reference_uncertainty_s):
    """Paired timestamps for the SAME independently timestamped sample/event.

    pc_reference_s must come from an independent trigger/reference clock,
    not a copy of the serial receive timestamp. Human review is required to
    validate this provenance; code cannot infer it from column names.
    """
    data=rows(points)
    values=np.array([[float(r[k]) for k in ('device_s','pc_reference_s','pc_receive_s')] for r in data])
    if len(values)<3 or not np.isfinite(values).all():raise ValueError('Three or more finite paired timing points required')
    device,reference,receive=values.T
    if np.any(np.diff(device)<=0) or np.any(np.diff(reference)<=0):raise ValueError('Nonmonotonic timing reference')
    if not np.isfinite(reference_uncertainty_s) or reference_uncertainty_s<0:raise ValueError('Reference uncertainty required')
    if np.any(receive<reference):raise ValueError('Receive cannot precede reference acquisition')
    latency=float(np.median(receive-reference))
    errors=receive-reference-latency
    fixed_scale_residual=reference-device-np.median(reference-device)
    save(output,dict(verified=False,method='calibrated_receive',receive_latency_s=latency,
         uncertainty_s=float(np.max(abs(errors))+np.max(abs(fixed_scale_residual))+reference_uncertainty_s),
         max_receive_jitter_s=float(np.max(abs(receive-device-np.median(receive-device)))),
         reference_uncertainty_s=reference_uncertainty_s,span_s=float(device[-1]-device[0]),
         source_sha256=digest(points),note='Pilot estimate, not a guaranteed bound. Validate sample/event matching, drift, transport settings and margin before marking verified.'))
