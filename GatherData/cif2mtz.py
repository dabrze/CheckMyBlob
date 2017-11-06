# !/usr/bin/python
import os, sys
import shutil
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

import config

from common import create_dirs
from common import replace_path
import ccp4
ccp4.setup()


def process_file(old_path, new_path):
    if not os.path.exists(new_path) and old_path.endswith('sf.ent'):
        print("{} -> {}".format(old_path, new_path))
        io = {
            'HKLIN' : old_path,
            'HKLOUT': new_path,
            }
        keywords = {
            #'SYMMETRY':'',
            #'CELL':'',
        }
        print('Running CIF2MTZ HKLIN {} HKLOUT {}'.format(old_path, new_path))
        cif2mtz = ccp4.CCP4_CIF2MTZ(io, keywords)
        cif2mtz.run()


def run(in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for root, dirs, files in os.walk(in_dir, topdown=True):
        create_dirs(root, out_dir, dirs)
         
        path = replace_path(root, out_dir)
        print root, out_dir, path

        for name in files:
            if name.endswith('sf.ent'):
                new_path = os.path.join(path, name.replace('sf.ent', '.mtz')[1:])
            old_path = os.path.join(root, name)
            
            process_file(old_path, new_path)

print config.cif2mtz_in_dir, config.cif2mtz_out_dir

run(config.cif2mtz_in_dir, config.cif2mtz_out_dir)
