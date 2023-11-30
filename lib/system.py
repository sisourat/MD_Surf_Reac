import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from dataclasses import dataclass
from utils import *

from utils import angtoau, evtoau

#from numba import njit

@dataclass
class System:
    D_a : float = 2.45 * evtoau
    Delta_a : float = 0.2
    z0 : float = 0.0
    alpha_a : float = 1.0  / angtoau
    D_m : float = 4.745 * evtoau
    Delta_m : float = -0.2
    r0 : float = 0.741 * angtoau
    alpha_m : float = 1.943  / angtoau

    m1 : float = 1.00794
    m2 : float = 1.00794
    M : float = 0.0
    mu : float = 0.0
    k_a : float = 0.0
    omega_a : float = 0.0

    def __post_init__(self):
        self.M = self.m1 + self.m2
        self.mu = self.m1*self.m2/self.M
        self.k_a = 2.0*self.D_a*self.alpha_a**2
        self.omega_a = np.sqrt(self.k_a / self.m1)

    def __copy__(self):
        new_obj = self.__class__()
        for k, v in vars(self).items():
            try:
                setattr(new_obj, k, copy.deepcopy(v))
            except:
                pass  # non-pickelable stuff wasn't needed

        return new_obj

    def copy(self):
        return self.__copy__()

    def Ua(self,x):
        return self.D_a / (4 * (1 + self.Delta_a)) * (
                (3 + self.Delta_a) * np.exp(-2 * self.alpha_a * (x - self.z0)) - (2 + 6 * self.Delta_a) * np.exp(
            -self.alpha_a * (x - self.z0)))

    def Um(self,x):
        return self.D_m / (4 * (1 + self.Delta_m)) * (
                    (3 + self.Delta_m) * np.exp(-2 * self.alpha_m * (x - self.r0)) - (2 + 6 * self.Delta_m) * np.exp(
                -self.alpha_m * (x - self.r0)))

    def Qa(self,x):
        return self.D_a / (4 * (1 + self.Delta_a)) * (
                    (1 + 3 * self.Delta_a) * np.exp(-2 * self.alpha_a * (x - self.z0)) - (6 + 2 * self.Delta_a) * np.exp(
                -self.alpha_a * (x - self.z0)))

    def Qm(self,x):
        return self.D_m / (4 * (1 + self.Delta_m)) * (
                    (1 + 3 * self.Delta_m) * np.exp(-2 * self.alpha_m * (x - self.r0)) - (6 + 2 * self.Delta_m) * np.exp(
                -self.alpha_m * (x - self.r0)))

    def pot3d(self,rho, z, Z):
        # rho
        # z = z2 - z1
        # Z = (m1*z1 + m2*z2)/M
        z1 = Z - (self.m2 / self.M) * z
        z2 = Z + (self.m1 / self.M) * z
        r = np.sqrt(rho * rho + z * z)
        return self.Ua(z1) + self.Ua(z2) + self.Um(r) - np.sqrt(self.Qm(r) ** 2 + (self.Qa(z1) + self.Qa(z2)) ** 2 - (self.Qa(z1) + self.Qa(z2)) * self.Qm(r))

    def Vfar_a(self,z1):
        return self.Ua(z1) - np.abs(self.Qa(z1))

    def Vfar_m(self,b):
        return self.Um(b) - np.abs(self.Qm(b))

    def plot_potA_potM(self,zstart,zstop,rstart,rstop,n):
        #plt.rcParams['interactive'] = True  ## ou sinon utiliser show()

        z = np.linspace(zstart, zstop, n)
        r = np.linspace(rstart, rstop, n)

        plt.figure(figsize=(10, 8))
        plt.axhline(0.0, c='k', lw=0.2)
        plt.plot(z, self.Ua(z), label='$U_a$')
        plt.plot(z, self.Qa(z), label='$Q_a$')
        plt.plot(z, self.Ua(z) - abs(self.Qa(z)), label='$U_a-|Q_a|$')
        plt.plot(r, self.D_a * (1 - np.exp(-self.alpha_a * (z - self.z0))) ** 2 - self.D_a, ':r', label='Morse a')
        plt.xlim(-1, 8)
        plt.ylim(-0.05, 0.0125)
        plt.xlabel('$z$')
        plt.ylabel('$V$')
        plt.legend()

        plt.figure(figsize=(10, 8))
        plt.axhline(0.0, c='k', lw=0.2)
        plt.plot(r, self.Um(r), label='$U_m$')
        plt.plot(r, self.Qm(r), label='$Q_m$')
        plt.plot(r, self.Um(r) - abs(self.Qm(r)), label='$U_m-|Q_m|$')
        plt.plot(r, self.D_m * (1 - np.exp(-self.alpha_m * (r - self.r0))) ** 2 - self.D_m, ':r', label='Morse m')
        plt.xlim(2.5, 10)
        plt.ylim(-.2, 1.25)
        plt.xlabel('$r$')
        plt.ylabel('$V$')
        plt.legend()
        plt.show()

# Fit du potential pour trouver la constante de raideur et le fond du puit
if __name__ == "__main__":
    mysys = System(m1=2.0)
    mysys.plot_potA_potM(zstart=-1 , zstop=10 ,rstart=-1 , rstop=10,n=1000)
    print(mysys.pot3d(rho = 1.0 ,z = 2.0 ,Z = 2.5 ))

