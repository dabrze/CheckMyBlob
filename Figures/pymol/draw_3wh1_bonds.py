import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append('D:\\VMs\kubuntu\VMSHARE\cmb_examples')
print(os.path.abspath(__file__))

from draw_script import *

bondslist = makeBonds(points_3wh1, r=0.05, color=oxygen)
cmd.load_cgo( bondslist, "bonds" )
