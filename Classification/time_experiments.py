# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>

import os
import logging
import subprocess
import threading
import time
import csv

import pandas as pd
from sklearn.externals import joblib

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

SEED = 23
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data'))
RESULTS_FOLDER = os.path.join(os.path.join(os.path.dirname(__file__), '..', 'Results'))
CAROLAN_PATH = os.path.join(DATA_FOLDER, "cl.pkl")
MAPS_PATH = os.path.join(DATA_FOLDER, "TimeExperimentMaps")
COORDINATES_FILE = os.path.join(MAPS_PATH, "sample.csv")


class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            self.process = subprocess.Popen(self.cmd, shell=True)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)

        try:
            self.process.kill()
        except:
            pass


def time_command(cmd, timeout):
    start = time.time()
    Command(cmd).run(timeout)
    end = time.time()

    elapsed = end - start

    if elapsed > timeout:
        return -1
    else:
        return elapsed


def run_warp(pdb, x, y, z, timeout):
    cmd = "$warpbin/auto_ligand.sh "
    cmd += "datafile {0}/{1}_refmac.mtz ".format(MAPS_PATH, pdb)
    cmd += "protein {0}/{1}_cut.pdb ".format(MAPS_PATH, pdb)
    cmd += "ligand {0}/arp_ligands/all_ligands.pdb ".format(MAPS_PATH)
    cmd += "search_position {0} {1} {2}".format(x, y, z)
    logging.info(cmd)

    return time_command(cmd, timeout)


def run_phenix(pdb, timeout):
    conf = "ligand_identification {"
    conf += """
    mtz_in = "{0}/{1}_refmac.mtz"
    mtz_type = F *diffmap
    input_labels = "DELFWT PHDELWT"
    ligand_dir = {0}/phenix_ligands/
    model = "{0}/{1}_cut.pdb"
    output_dir = "{0}"
    job_title = "{1}"
    """.format(MAPS_PATH, pdb)
    conf += """}
"""

    conf_file_name = "{0}/{1}.eff".format(MAPS_PATH, pdb)
    with open(conf_file_name, "w") as conf_file:
        conf_file.write(conf)

    cmd = "phenix.ligand_identification {0}".format(conf_file_name)
    logging.info(cmd)

    return time_command(cmd, timeout)


def run_cmb(pdb, timeout):
    return 0


def write_result(pdb, method, time, save_to_folder=RESULTS_FOLDER, file_name="TimeComparison.csv"):
    if not os.path.exists(save_to_folder):
        os.mkdir(save_to_folder)
    file_path = os.path.join(save_to_folder, file_name)

    if os.path.isfile(file_path):
        write_header = False
        mode = "a"
    else:
        write_header = True
        mode = "w"

    with open(file_path, mode) as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')

        if write_header:
            header = ["pdb",
                      "method",
                      "time"]
            writer.writerow(header)

        row = [pdb, method, time]
        writer.writerow(row)


def measure_time(pdb, x, y, z, timeout=3600):
    t = run_warp(pdb, x, y, z, timeout)
    write_result(pdb, "cl", t)

    t = run_phenix(pdb, timeout)
    write_result(pdb, "tamc", t)
    #
    # t = run_cmb(pdb, timeout)
    # write_result(pdb, "cmb", t)


if __name__ == '__main__':
    warp_errors = ["4m8u", "4pn9", "4cgs"]

    data = joblib.load(CAROLAN_PATH)
    sample = data.data_frame.sample(n=33, random_state=SEED).index.str[0:4].values
    sample = [x for x in sample if x not in warp_errors]
    coordinates = pd.read_csv(COORDINATES_FILE, index_col=0)

    for pdb in sample:
        c = coordinates[coordinates.pdb == pdb]
        measure_time(pdb, float(c.x), float(c.y), float(c.z))
