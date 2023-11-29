import json
from system import *

# Opening JSON file
f = open('input.json')
# returns JSON object as
# a dictionary
data = json.load(f)
# Closing file
f.close()

print(data['potentialA']['D_a'])
mysys = System(D_a=data['potentialA']['D_a'] * u.ev,
               Delta_a=data['potentialA']['Delta_a'],
               alpha_a=data['potentialA']['alpha_a'] * u.angstrom**(-1),
               D_m=data['potentialM']['D_m'] * u.ev,
               Delta_m=data['potentialM']['Delta_m'],
               alpha_m=data['potentialM']['alpha_m'] * u.angstrom**(-1),
               m1=data['masses']['m1'] * u.amu,
               m2=data['masses']['m2'] * u.amu
)

print(mysys.pot3d(rho = 1.0 * u.angstrom,z = 2.0 * u.angstrom,Z = 2.5 * u.angstrom))