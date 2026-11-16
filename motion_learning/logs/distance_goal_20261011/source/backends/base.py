"""SI-unit state contract for future Gym and alternative simulator adapters."""
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class RobotState:
    time_s: float
    joint_position_rad: Dict[str, float]
    joint_velocity_rad_s: Dict[str, float]
    rpy_rad: Tuple[float, float, float]
    gyro_rad_s: Tuple[float, float, float]
    acceleration_m_s2: Tuple[float, float, float]
    height_m: float  # Supervisor truth: diagnostics only, not a hardware sensor.
