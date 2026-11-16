"""Explicit session QC amendment; preserve immutable calibration and raw data."""
from pathlib import Path
import contextlib, copy, json, math
import repeat_analysis as frozen_analysis

_original_reconstruct = frozen_analysis.reconstruct
ORIGINAL_TIMING_ISSUE = 'Power timing exceeds the original 0.003 s jitter bound'


def load_policy(session, cfg=None):
    session = Path(session)
    path = session/'quality_policy.json'
    if path.exists():
        policy = json.loads(path.read_text(encoding='utf-8-sig'))
        limit = policy.get('max_receive_jitter_s')
        if (policy.get('schema_version') != 1 or not policy.get('policy_id') or
                policy.get('user_requested') is not True or isinstance(limit, bool) or
                not isinstance(limit, (float, int)) or not math.isfinite(limit) or
                limit not in (.003, .005)):
            raise ValueError('品質基準ファイルが不正です。収録・合格判定を停止します。')
        return policy
    limit = cfg['timing']['max_receive_jitter_s'] if cfg else .003
    return {'schema_version':1, 'policy_id':'original_recording_config',
            'max_receive_jitter_s':limit, 'user_requested':False,
            'protocol_amended':False}


def policy_for_raw(raw, cfg):
    raw = Path(raw)
    # Session/trials/attempt/raw is the live console capture layout. Standalone
    # archived reference data and simulated fixtures retain their own config.
    if len(raw.parents)>2 and raw.parent.parent.name=='trials':
        return load_policy(raw.parents[2], cfg)
    return {'schema_version':1, 'policy_id':'original_recording_config',
            'max_receive_jitter_s':cfg['timing']['max_receive_jitter_s'],
            'user_requested':False, 'protocol_amended':False}


def apply_policy(qc, original_limit, policy):
    result = copy.deepcopy(qc)
    original_issues = list(qc['quality_issues'])
    limit = policy['max_receive_jitter_s']
    issues = [reason for reason in original_issues if reason != ORIGINAL_TIMING_ISSUE]
    if qc['power_jitter_s'] > limit:
        issues.append(f'Power timing exceeds active {limit:.6f} s jitter bound')
    result.update(quality_issues=issues, power_jitter_limit_s=limit,
                  quality_policy_id=policy['policy_id'],
                  original_power_jitter_limit_s=original_limit,
                  original_quality_issues=original_issues,
                  original_quality_pass=not original_issues,
                  protocol_amended=(limit != original_limit))
    return result


def reconstruct(raw, support, strict=True, policy=None):
    raw = Path(raw)
    cfg = json.loads((raw/'config.json').read_text(encoding='utf-8-sig'))
    data, qc = _original_reconstruct(raw,support,strict=False)
    policy = policy if policy is not None else policy_for_raw(raw,cfg)
    qc = apply_policy(qc,cfg['timing']['max_receive_jitter_s'],policy)
    if strict and qc['quality_issues']:
        raise ValueError('; '.join(qc['quality_issues']))
    return data,qc


@contextlib.contextmanager
def session_evaluation_policy(session):
    """Use one policy snapshot for current session analysis, then restore it."""
    root = (Path(session)/'trials').resolve()
    policy = load_policy(session)
    previous = frozen_analysis.reconstruct
    def scoped(raw,support,strict=True):
        try:Path(raw).resolve().relative_to(root)
        except ValueError:return previous(raw,support,strict)
        return reconstruct(raw,support,strict,policy)
    frozen_analysis.reconstruct = scoped
    try:yield policy
    finally:frozen_analysis.reconstruct = previous
