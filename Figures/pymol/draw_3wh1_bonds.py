import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pymol_data'))
print(os.path.abspath(__file__))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pymol_data')))

from draw_script import *

bondslist = makeBonds(points_3wh1, r=0.05, color=oxygen)
cmd.load_cgo( bondslist, "bonds" )
