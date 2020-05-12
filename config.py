from enum import Enum
from dataclasses import dataclass

class ForceConstraint:
    NO_PUDDLE = 0
    MEAN_CONSTRAINED = 1
    CHANCE_CONSTRAINED = 2

@dataclass
class PhysicalParams:
    lf : float
    lr : float
    m : float
    Iz : float
    wheel_length : float
    wheel_width : float
    
@dataclass
class Limits:
    min_xdot : float
    min_slip_ratio_mag : float
    max_ddelta : float
    max_dkappa : float
    max_delta : float