
from system import *
from read_input import *


data = read_input('input.json')
print(data)
mysys = System(D_a=data['D_a'] ,
               Delta_a=data['Delta_a'],
               alpha_a=data['alpha_a'] ,
               D_m=data['D_m'] ,
               Delta_m=data['Delta_m'],
               alpha_m=data['alpha_m'] ,
               m1=data['m1'] ,
               m2=data['m2'])

print(data["blist"])

print(mysys.pot3d(rho = 1.0 ,z = 2.0 ,Z = 2.5))