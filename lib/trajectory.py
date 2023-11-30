import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.integrate as si
from dataclasses import dataclass

from utils import angtoau, evtoau, fstoau

@dataclass
class Trajectory:
    tmax : float = 2000.0
    dt : float = 0.001
    dz : float = 0.001
    dZ : float = 0.001
    drho : float = 0.001
    ecoll : float = 1.0


############################################################################################################################

    def initial_gen_target(z_i, K_i, freq, mass,
                           number):  # Generate a number of initial conditions (z1,vz1)_0 for target atom
        vz_i = np.sqrt(2 * K_i / mass)
        stdz = np.sqrt(1 / (2 * freq * mass))
        stdv = np.sqrt((2 * freq) / mass)
        z_0 = np.random.normal(z_i, stdz, number)
        vz_0 = (2 * np.random.randint(0, 2, number) - 1) * np.random.normal(vz_i, stdv, number)
        return z_0, vz_0


    def initial_gen_proj(z2_i, b, theta, K_i, Delta_rho, Delta_K, mass,
                         number):  # Generate a number of initial conditions (z2, rho, vz2, vrho)_0 for projectile atom
        z_0 = z2_i * np.ones(number)

        stdrho = Delta_rho / 2
        rho_0 = np.random.normal(b, stdrho, number)

        stdK = Delta_K / 2
        K_0 = np.abs(np.random.normal(K_i, stdK,
                                      number))  # Absolute value to avoid negative Kinetic energies when K_i<Delta_K (Non gaussian distrib tho)
        v_0 = np.sqrt(2 * K_0 / mass)
        vz_0 = - v_0 * np.cos(theta)
        vrho_0 = - v_0 * np.sin(theta)
        return z_0, rho_0, vz_0, vrho_0

############################################################################################################################


    def diff_sys(y0, t, drho, dz, dZ):
        # Defines the Differential system that will be solved to find the trajectory
        # Takes y0 = (pos,vit) and computes its derivative dy0/dt = (vit,acc)
        rho, z, Z, vrho, vz, vZ = y0
        Arho = -(pot.V(rho + .5 * drho, z, Z) - pot.V(rho - .5 * drho, z, Z)) / (inp.mu * drho)
        Az = -(pot.V(rho, z + .5 * dz, Z) - pot.V(rho, z - .5 * dz, Z)) / (inp.mu * dz)
        AZ = -(pot.V(rho, z, Z + .5 * dZ) - pot.V(rho, z, Z - .5 * dZ)) / (inp.M * dZ)
        return np.array([vrho, vz, vZ, Arho, Az, AZ])

############################################################################################################################

    def prop_traj(z2_0, vz2_0, rho_0, vrho_0, z1_0, vz1_0, t):
        Recomb = 0

        rho_in = rho_0
        z_in = z2_0 - z1_0
        Z_in = (inp.m2 * z2_0 + inp.m1 * z1_0) / inp.M
        vrho_in = vrho_0
        vz_in = vz2_0 - vz1_0
        vZ_in = (inp.m2 * vz2_0 + inp.m1 * vz1_0) / inp.M

        y0 = np.array([rho_in, z_in, Z_in, vrho_in, vz_in, vZ_in])
        y = si.odeint(diff_sys, y0, t, args=(inp.drho, inp.dz, inp.dZ))

        rho, z, Z, vrho, vz, vZ = y.T
        z1 = Z - (inp.m2 / inp.M) * z
        z2 = Z + (inp.m1 / inp.M) * z
        r = np.sqrt(rho * rho + z * z)

        r_fin = r[-1]
        if r_fin < inp.r_rec:
            Recomb = 1

        return (y, Recomb)

    def data_traj(y, t):
        rho, z, Z, vrho, vz, vZ = y.T
        z1 = Z - (inp.m2 / inp.M) * z
        z2 = Z + (inp.m1 / inp.M) * z
        r = np.sqrt(rho * rho + z * z)

        KZ = 0.5 * inp.M * vZ * vZ
        Kr = 0.5 * inp.mu * (vz * vz + vrho * vrho)
        V = pot.V(rho, z, Z)

        return (rho, z, Z, z1, z2, r, KZ, Kr, V)


