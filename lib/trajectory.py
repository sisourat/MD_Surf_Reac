import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from dataclasses import dataclass
from utils import *

from pint import UnitRegistry

u = UnitRegistry()
u.define('ev = 1.602176634e-19 joule')



@dataclass
class Trajectory:
    tmax : float = 0.0 * u.femtoseconds
    dt : float = 0.024 * u.femtoseconds
    dz : float = 0.001 * u.angstrom
    dZ : float = 0.001 * u.angstrom
    drho : float = 0.001 * u.angstrom

    ecoll : float = 0.001 * u.angstrom

