import json
from module_classes import *

# Opening JSON file
f = open('input.json')
# returns JSON object as
# a dictionary
data = json.load(f)
# Closing file
f.close()

mysys = System(D_a=data['D_a'] * u.ev)


