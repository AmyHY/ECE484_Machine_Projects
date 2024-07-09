from enum import Enum, auto
import copy
from typing import List

class PedestrianMode(Enum):
    Normal=auto()

class VehicleMode(Enum):
    Normal = auto()
    Brake = auto()
    Accel = auto()
    HardBrake = auto()

class State:
    x: float 
    y: float 
    theta: float 
    v: float 
    agent_mode: VehicleMode 

    def __init__(self, x, y, theta, v, agent_mode: VehicleMode):
        pass 

def decisionLogic(ego: State, other: State):
    output = copy.deepcopy(ego)
    
    if ego.agent_mode == VehicleMode.Normal and other.dist < 60 and other.dist >= 26:
        output.agent_mode = VehicleMode.Accel
    if ego.agent_mode == VehicleMode.Accel and other.dist < 26 and other.dist >= 25:
        output.agent_mode = VehicleMode.Normal
    if ego.agent_mode == VehicleMode.Normal and other.dist < 25 and other.dist >= 3:
        output.agent_mode = VehicleMode.Brake
    if ego.agent_mode == VehicleMode.Brake and other.dist < 3:
        output.agent_mode = VehicleMode.HardBrake
    if ego.agent_mode == VehicleMode.HardBrake or ego.agent_mode == VehicleMode.Brake and other.dist > 
26:
        output.agent_mode = VehicleMode.Normal 

    assert other.dist > 2.0

    return output 
