import sys

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.integrate as si
from dataclasses import dataclass, field
from icecream import ic

from utils import angtoau, evtoau, fstoau

@dataclass
class Dynamic:
    tmax : float = 2000.0
    dt : float = 0.001
    dz : float = 0.001
    dZ : float = 0.001
    drho : float = 0.001
    ecoll : float = 1.0

    nb : float = 0
    blist : list = field(default_factory=list)
    ntraj : int = 20

    tlist : list = field(default_factory=list)

    zt_i : float = 0.0
    ekt_i : float = 0.0

    zp_i : float = 0.0
    ekp_i : float = 0.0
    theta : float = 0.0  # normal incidence

    delta_rho : float = 0.0
    delta_ek : float = 0.0

############################################################################################################################
    @staticmethod
    def initial_gen_target(zt, ekt, freq, mass, number):
        # Generate a number of initial conditions (zt,vzt)_0 for target atom
        vz = np.sqrt(2.0 * ekt / mass)
        stdz = np.sqrt(1.0 / (2.0 * freq * mass))
        stdv = np.sqrt((2.0 * freq) / mass)
        z_0 = np.random.normal(zt, stdz, number)
        vz_0 = (2 * np.random.randint(0, 2, number) - 1) * np.random.normal(vz, stdv, number)
        return z_0, vz_0

    @staticmethod
    def initial_gen_proj(zp, b, theta, ekp, delta_rho, delta_ek, mass, number):
        # Generate a number of initial conditions (z2, rho, vz2, vrho)_0 for projectile atom
        z_0 = zp * np.ones(number)

        stdrho = delta_rho / 2.0
        rho_0 = np.random.normal(b, stdrho, number)

        stdek = delta_ek / 2.0
        ek_0 = np.abs(np.random.normal(ekp, stdek, number))
        # Absolute value to avoid negative Kinetic energies when ek_i<delta_ek (Non gaussian distrib tho)
        v_0 = np.sqrt(2 * ek_0 / mass)
        vz_0 = - v_0 * np.cos(theta)
        vrho_0 = - v_0 * np.sin(theta)
        return z_0, rho_0, vz_0, vrho_0

############################################################################################################################

    @staticmethod
    def diff_sys(y0, tlist, pot, mu, mtot, drho, dz, dZ):
        # Defines the Differential system that will be solved to find the trajectory
        # Takes y0 = (pos,vit) and computes its derivative dy0/dt = (vit,acc)
        rho, z, Z, vrho, vz, vZ = y0
        Arho = -(pot(rho + .5 * drho, z, Z) - pot(rho - .5 * drho, z, Z)) / (mu * drho)
        Az = -(pot(rho, z + .5 * dz, Z) - pot(rho, z - .5 * dz, Z)) / (mu * dz)
        AZ = -(pot(rho, z, Z + .5 * dZ) - pot(rho, z, Z - .5 * dZ)) / (mtot * dZ)
        return np.array([vrho, vz, vZ, Arho, Az, AZ])

############################################################################################################################

    def solv_traj(self, sys, dyn, zp_0, vzp_0, rho_0, vrho_0, zt_0, vzt_0):
        t = dyn.tlist
        recomb = 0

        rho_in = rho_0
        z_in = zp_0 - zt_0
        Z_in = (sys.mp * zp_0 + sys.mt * zt_0) / sys.mtot
        vrho_in = vrho_0
        vz_in = vzp_0 - vzt_0
        vZ_in = (sys.mp * vzp_0 + sys.mt * vzt_0) / sys.mtot

        #ic(vzp_0,vzt_0)
        #ic(rho_in, z_in, Z_in, vrho_in, vz_in, vZ_in)

        y0 = np.array([rho_in, z_in, Z_in, vrho_in, vz_in, vZ_in])
        y = si.odeint(self.diff_sys, y0, t, args=(sys.pot3d, sys.mu, sys.mtot, dyn.drho, dyn.dz, dyn.dZ))
    #   print(y[-1,:])

       # plt.plot(t, y[:, 0], 'b', label='rho(t)')
       # plt.plot(t, y[:, 1], 'g', label='z(t)')
       # plt.plot(t, y[:, 2], 'r', label='Z(t)')
       # plt.legend(loc='best')
       # plt.xlabel('t')
       # plt.grid()
       # plt.show()
       # sys.exit()

        rho, z, Z, vrho, vz, vZ = y.T
        z1 = Z - (sys.mp / sys.mtot) * z
        z2 = Z + (sys.mt / sys.mtot) * z
        r = np.sqrt(rho * rho + z * z)

        r_fin = r[-1]
        ic(r_fin)
        #sys.exit()
        if r_fin < sys.r_rec:
            recomb = 1

        return (y, recomb)

    @staticmethod
    def data_traj(sys, y, t):
        rho, z, Z, vrho, vz, vZ = y.T
        zt = Z - (sys.mp / sys.mtot) * z
        zp = Z + (sys.mt / sys.mtot) * z
        r = np.sqrt(rho * rho + z * z)

        eZ = 0.5 * sys.mtot * vZ * vZ
        er = 0.5 * sys.mu * (vz * vz + vrho * vrho)
        pot = sys.pot3d(rho, z, Z)

        return (rho, z, Z, zt, zp, r, eZ, er, pot)


    def solv_ntraj(self, sys, dyn, zp_0_vect, vzp_0_vect, rho_0_vect, vrho_0_vect, zt_0_vect, vzt_0_vect):

        ecoll = dyn.ecoll
        t = dyn.tlist
        ntraj = dyn.ntraj
        n_rec = 0

        eZ_fin = np.zeros(ntraj)
        er_fin = np.zeros(ntraj)
        evib_fin = np.zeros(ntraj)

        for i in range(ntraj):
            zt_0 = zt_0_vect[i]
            vzt_0 = vzt_0_vect[i]

            zp_0 = zp_0_vect[i]
            rho_0 = rho_0_vect[i]
            vzp_0 = vzp_0_vect[i]
            vrho_0 = vrho_0_vect[i]
            y, recomb = self.solv_traj(sys, dyn, zp_0, vzp_0, rho_0, vrho_0, zt_0, vzt_0)

            n_rec += recomb

            rho, z, Z, zt, zp, r, eZ, er, V = self.data_traj(sys, y, t)

            eZ_fin[i] = eZ[-1] * recomb
            er_fin[i] = er[-1] * recomb
            evib_fin[i] = (er[-1] + V[-1] - ecoll + sys.D_m) * recomb

        if n_rec == 0:
            eZ_fin_avg = 0.
            er_fin_avg = 0.
            evib_fin_avg = 0.
        else:
            eZ_fin_avg = eZ_fin.sum() / n_rec
            er_fin_avg = er_fin.sum() / n_rec
            evib_fin_avg = evib_fin.sum() / n_rec

        return (n_rec, eZ_fin_avg, er_fin_avg, evib_fin_avg)
