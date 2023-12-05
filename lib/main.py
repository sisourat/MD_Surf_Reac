
from system import *
from read_input import *
from  dynamic import *


data = read_input('input.json')
#print(data)
mysys = System(D_a=data['D_a'] ,
               Delta_a=data['Delta_a'],
               alpha_a=data['alpha_a'] ,
               D_m=data['D_m'] ,
               Delta_m=data['Delta_m'],
               alpha_m=data['alpha_m'] ,
               m1=data['m1'] ,
               m2=data['m2'])

mydyn = Dynamic(tmax=data['tmax'], dt=data['dt'], dz=data['dz'], dZ=data['dZ'],
                drho=data['drho'], ntraj=data['ntraj'], blist=data['blist'], ecoll=data['ecoll'],
                zp_i=data['zp_i'], tlist=data['tlist'])

theta=0.0 # normal incidence
p_rec = np.zeros(len(mydyn.blist))
delta_p_rec = np.zeros(len(mydyn.blist))
for ib, b in enumerate(mydyn.blist):
   zt_0_vect, vzt_0_vect = mydyn.initial_gen_target(mydyn.zt_i,mydyn.ekt_i,mysys.omega_a,mysys.m1,mydyn.ntraj)
   zp_0_vect, rho_0_vect, vzp_0_vect, vrho_0_vect = mydyn.initial_gen_proj(mydyn.zp_i,b,mydyn.theta,mydyn.ecoll
                                                    ,mydyn.delta_rho,mydyn.delta_ek,mysys.m2,mydyn.ntraj)
   n_rec, eZ_fm, er_fm, evib_fm = mydyn.solv_ntraj(mysys, mydyn, zp_0_vect, vzp_0_vect, rho_0_vect, vrho_0_vect,
                                                   zt_0_vect, vzt_0_vect)

   p_rec[ib] = n_rec / mydyn.ntraj
   if n_rec == 0:
       delta_p_rec[ib] = 0
   else:
       delta_p_rec[ib] = n_rec / mydyn.ntraj * np.sqrt((mydyn.ntraj - n_rec) / (mydyn.ntraj * n_rec))

#print(mydyn.blist)
#print(zt_0_vect,vzt_0_vect)
#print(zp_0_vect,vzp_0_vect)

#print(mysys.pot3d(rho = 1.0 ,z = 2.0 ,Z = 2.5))