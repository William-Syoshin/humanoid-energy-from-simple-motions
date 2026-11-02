"""Deterministically instrument the pinned official model without changing physics."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SHA256 = 'dec8ae7a6d6a99b2e9025effd2c4dadd92b482db39bb6e95152a2b79fb85ad58'
FIXED_JOINTS = ('LeftElbow', 'RightElbow', 'Waist', 'LeftFemurHead1', 'RightFemurHead1')
COMMIT = 'c6793d8f7230a311c4bc2a3101d9f1a8bc0aa01b'


def matching_brace(text, opening):
    depth = 0
    for i in range(opening, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError('Unbalanced PROTO')


def fix_joint(text, name):
    # The pinned source has zero initial joint positions and no joint rotations.
    # Keep the endPoint Solid (geometry, mass, descendants), rigidly attached to
    # its parent. No virtual servo or freely swinging passive hinge remains.
    marker = 'DEF KHR3HV_' + name + ' HingeJoint {'
    start = text.index(marker)
    end = matching_brace(text, text.index('{', start))
    endpoint = text.index('endPoint Solid {', start, end)
    solid_start = endpoint + len('endPoint ')
    solid_end = matching_brace(text, text.index('{', solid_start))
    return text[:start] + text[solid_start:solid_end + 1] + text[end + 1:]


def build():
    source = (ROOT / 'vendor/webots/R2025a/Khr3hv.proto').read_bytes()
    if hashlib.sha256(source).hexdigest() != SHA256:
        raise ValueError('Pinned upstream PROTO checksum mismatch')
    text = source.decode()
    for name in FIXED_JOINTS:
        text = fix_joint(text, name)
    motors = re.findall(r'RotationalMotor\s*\{[^{}]*\}', text)
    joints = []
    for block in motors:
        name = re.search(r'name\s+"([^"]+)"', block)[1]
        def number(field):
            return float(re.search(r'\b' + field + r'\s+([-\d.]+)', block)[1])
        joints.append(dict(name=name, sensor=name + '_sensor',
                           model_min_rad=number('minPosition'),
                           model_max_rad=number('maxPosition'),
                           model_max_velocity_rad_s=number('maxVelocity'),
                           model_max_torque_nm=number('maxTorque')))
    assert len(joints) == 17 and 'PositionSensor {' not in text
    def instrument(match):
        indent, block = match.group(1), match.group(2)
        name = re.search(r'name\s+"([^"]+)"', block)[1]
        return (indent + block + '\n' + indent + 'PositionSensor {\n' +
                indent + '  name "' + name + '_sensor"\n' + indent + '}')
    text = re.sub(r'(?m)^(\s*)(RotationalMotor\s*\{[^{}]*\})', instrument, text)
    text = text.replace('PROTO Khr3hv [', 'PROTO Khr3hv17 [', 1)
    text = text.replace('# keywords:', '# Derived locally: 5 expansion hinges fixed at zero; 17 PositionSensors added; masses unchanged.\n# keywords:', 1)
    # Share unchanged texture assets with the archived upstream source.
    text = text.replace('"textures/', '"../vendor/webots/R2025a/textures/')
    (ROOT / 'protos/Khr3hv17.proto').write_text(text)
    manifest = dict(webots_release='R2025a', upstream_commit=COMMIT,
                    upstream_proto_sha256=SHA256, joints=joints,
                    upstream_motor_count=22, upstream_sensor_count=0,
                    fixed_expansion_joints=list(FIXED_JOINTS),
                    added_body_devices={'imu': 'InertialUnit', 'gyro': 'Gyro',
                                        'accelerometer': 'Accelerometer'})
    (ROOT / 'config/model_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared 17 motors + 17 added position sensors; 5 expansion joints rigidly fixed; body sensors come from world bodySlot.')


if __name__ == '__main__':
    build()
