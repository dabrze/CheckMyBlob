import os
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# path to rsynced PDB repository with unizpped files
#pdb_repository_unzipped_dir = os.path.join(PATH, 'PDB')
pdb_repository_unzipped_dir = '/home/ubuntu/pdb'
pdb_structures_dir = os.path.join(pdb_repository_unzipped_dir, 'structures', 'all')

# path where output will end up
pdb_out_data_dir = os.path.join(PATH, 'Output')
cif2mtz_in_dir = os.path.join(pdb_structures_dir, 'structure_factors')
cif2mtz_out_dir = os.path.join(pdb_structures_dir, 'mtz')

# path to ccp4 setup script
ccp4_setup_path = '/usr/local/xtalprogs/setup.py'

