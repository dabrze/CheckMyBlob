import sys, os
sys.path.append(os.path.dirname(__file__))
print(os.path.abspath(__file__))

from draw_script import *

spherelist = makeSpheres(points_3wh1, r=0.14, color=brightorange)
cmd.load_cgo(spherelist, 'spheres',   1)
