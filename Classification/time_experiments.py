# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>

import os
import logging
import subprocess
import threading
import time
from sklearn.externals import joblib

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

SEED = 23
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data'))
RESULTS_FOLDER = os.path.join(os.path.join(os.path.dirname(__file__), '..', 'Results'))
CAROLAN_PATH = os.path.join(DATA_FOLDER, "cl.pkl")
MAPS_PATH = os.path.join(DATA_FOLDER, "TimeExperimentMaps")
COORDINATES_FILE = os.path.join(DATA_FOLDER, "TimeExperimentMaps", "sample.csv")
RESULTS_PATH = os.path.join(RESULTS_FOLDER, "TimeComparison.csv")


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
    cmd += "ligand /home/student/work/ccp4/arp_warp_7.6/share/all_ligands.pdb "
    cmd += "search_position {0} {1} {2}".format(x, y, z)
    logging.info(cmd)

    return time_command(cmd, timeout)


def run_phenix(pdb, timeout):
    conf = "ligand_identification {"
    conf += """
    mtz_in = "{0}/{1}_refmac.mtz"
    mtz_type = F *diffmap
    input_labels = "DELFWT PHDELWT"xcvvcxv
    ligand_list = SO4 GOL HEM PEG MPD FAD NAD PG4 NAP ADP MES TRS FMN "1PE" EPE \\
                  PLP SF4 ATP IMD CIT COA GSH AMP P6G LDA H4B NCO MLI SIA TLA \\
                  SAM DIO AKG ACO PYR PLM BTB OLA TPP CYC POP CLA CHD BCL DTT \\
                  NHE PEP ADN PGO RET HED MYR F3S THP BTN CMP "2PE" ADE "5GP" \\
                  IPH CXS C2E B3P MLT TYD UPG CAM HEA A3P CDL TAM ORO PGA "2GP" \\
                  HC4 MTE U10 "017" STU PHQ FPP SPO
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


def measure_time(pdb, x, y, z, timeout=3600):
    print(run_warp(pdb, x, y, z, timeout))
    print(run_phenix(pdb, timeout))
    print(run_cmb(pdb, timeout))


if __name__ == '__main__':
    data = joblib.load(CAROLAN_PATH)
    sample = data.data_frame.sample(n=30, random_state=SEED).index.values

    for pdb in sample:
        print pdb
