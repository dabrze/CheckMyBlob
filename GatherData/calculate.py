#!/usr/bin/env python

import os
import sys
import math
from itertools import izip
import logging
import platform
import datetime
from collections import OrderedDict
import operator
import json
import hashlib
import argparse

import numpy as np
from scipy import ndimage

try:
    MATPLOTLIB = True
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except Exception as err:
    print "MATPLOT LIB IMPOR ERR"
    print err
    MATPLOTLIB = False

from skimage import morphology
from skimage import segmentation
from skimage import feature
import pandas as pd

# local
from ignored_res import IGNORED_RESIDUES
from ignored_res import KEEP_RESIDUES
from ignored_res import ELEMENTS_ELECTRONS
from ligand_dict import get_ligand_atoms_dict
from pyclipper import pyclipper

from moment_invariants import MomentCache
from moment_invariants import GeometricalInvariantCache

from zernike import ZernikeCoefficient
from zernike import ZernikeMomentCache

from blob import Blob
from utils import print_3d_mesh
from utils import drange
from utils import binormal
from utils import normal
from utils import fit_normal
from utils import fit_binormal

from mmLib import FileIO
from mmLib import ConsoleOutput

ConsoleOutput.disable()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

import config
import ccp4

ccp4.setup(config.ccp4_setup_path)
np.set_printoptions(precision=3)

class RefmacError(Exception):
    pass


class Options(object):
    def __init__(self):
        self.DEBUG_MORE = False
        self.DEBUG_MORE_TITLE_CONTAINS = ""
        self.BORDER = 1.5  # angstrom
        self.MASK_BORDER = 2.6  # angstrom
        self.MAX_RHO_BORDER = 2.6  # angstrom
        self.NEAR_ATOM_DIST = self.MASK_BORDER  # angstrom
        self.ORTH_GRID_SPACE = 0.2  # angstrom
        self.CLIPPER_GRID_SIZE_OTHER = 0.2  # self.ORTH_GRID_SPACE
        self.CLIPPER_GRID_SIZE_FOFC = 0.2  # self.ORTH_GRID_SPACE
        self.BLOB_MERGE_DIST = 2.15  # angstrom
        self.BLOB_GRAPH_DIST = 1.85 * self.ORTH_GRID_SPACE  # angstrom
        # self.BLOB_GRAPH_DIST_HI = self.BLOB_MERGE_DIST  # angstrom
        self.DENSITY_STD_THRESHOLD = 2.8  # 2.8 - default  # 5.7 # of diff std for PHWT or 3.5 for Fobs
        self.PART_STEP_COUNT = 2  # default 9
        self.PART_STEP_SIZE = 0.5
        self.PRINT_SLICE = False and self.DEBUG_MORE
        self.PRINT_3D_MESH = True and self.DEBUG_MORE
        self.PRINT_3D_CUT_MASK = True and self.DEBUG_MORE
        # the limit for cython
        self.MIN_VOLUME_RADIUS = 0.39  # 0.55
        self.MIN_VOLUME_LIMIT = 4.0 / 3.0 * 3.14 * ((self.MIN_VOLUME_RADIUS) ** 3)
        # the limit after merging
        self.MIN_VOLUME_RADIUS_MERGED_BLOB = 0.80  # 1.05
        self.MIN_VOLUME_LIMIT_MERGED_BLOB = 4.0 / 3.0 * 3.14 * ((self.MIN_VOLUME_RADIUS_MERGED_BLOB) ** 3)
        self.B_TO_U = 1.0 / (8.0 * math.pi * math.pi)
        self.SAVE_CUT_MAP = False
        self.STEPS = 15
        self.SHOW_DECAY = True
        self.SOLVENT_STATS = False
        self.SOLVENT_PLOT = False  # historgrams
        self.SOLVENT_RADIUS = 1.9
        self.SOLVENT_OPENING_RADIUS = 1.4
        self.BLOB = True  # False
        self.DIFF_MAP = False  # historgrams
        self.MAP_F = 'DELFWT'  # 'FWT'  #  #'FP'
        self.MAP_PH = 'PHDELWT'  # 'PHWT'  #  #'PHIC'
        self.MAP_W = ''
        self.MAX_RESOLUTION_LIMIT = 1.0  # 1.7
        # zwieksze nie RHO powoduje mniejsze wyciecie
        self.RHO = 0.30  # was 0.21
        self.PEAK_MAX_DISTACNE = 1  # 2
        self.MINIMAL_SEGMENT_PROBABILITY = 2.0
        self.DEFAULT_SPHERE_RADIUS = 1.05  # 0.85 , 1.65
        self.WALKER_MIN_SEGMENTATION_LIMIT = -80
        self.USE_TWO_FO_FC_MAP = False
        self.USE_FO_MAP = False
        self.USE_FC_MAP = False
        self.USE_FO_FC_MAP = False
        self.EDSTATS = False
        self.REFMAC_NCYC = 5  # 0
        self.REFMAC_WEIGHT = 'AUTO'  # 'AUTO'
        self.REFMAC_BFAC_SET = None  # maybe set to 20

        if self.DEBUG_MORE:
            logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


class DataAggregator(OrderedDict):
    def save_results(self, filename):
        out_file = open(filename, 'a')
        txt = ';'.join(("%s:%s" % (
            key,
            "%.10g" % value if isinstance(value, float) else str(value)) for (key, value) in self.iteritems())
                       )
        print >> out_file, txt
        out_file.close()

    def print_results(self):
        txt = ';'.join(("%s:%s" % (
            key,
            "%.10g" % value if isinstance(value, float) else str(value)) for (key, value) in self.iteritems())
                       )
        print txt


class GatherData(object):
    def __init__(self, pdb_code, pdb_file, cif_file, mtz_file, options, logger, pdb_out_data_dir=None, overwrite=False,
                 rerefine=True):
        self.logger = logger
        self.pdb_code = pdb_code
        self.options = options
        self.rerefine = rerefine

        if pdb_out_data_dir is not None:
            config.pdb_out_data_dir = pdb_out_data_dir

        self.dir_static = {
            'repo': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
            'cif': os.path.join(config.pdb_repository_unzipped_dir, 'structures', 'all', 'mmCIF'),
            'map': os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'map'),  # not used
            'log': os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'log'),  # not used
            'temp': os.path.join(config.pdb_out_data_dir, 'tmp'),
            'mtz': os.path.join(config.pdb_repository_unzipped_dir, 'structures', 'all', 'mtz'),
            'pdb': os.path.join(config.pdb_repository_unzipped_dir, 'structures', 'all', 'pdb'),
            'edstats': os.path.join(config.pdb_repository_unzipped_dir, 'structures', 'all', 'edstats'),
        }

        self.directories = {
            'temp': os.path.join(self.dir_static['temp'], pdb_code),
            'data': os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'all'),
            'graphs': os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'graphs'),
            'classifier': os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'classifier'),
            '2FoFc_solvent_histogram': os.path.join(config.pdb_out_data_dir, 'checkmyblob', '2FoFc', 'solvent_hist'),
            '2FoFc_atom_histogram': os.path.join(config.pdb_out_data_dir, 'checkmyblob', '2FoFc', 'atom_hist'),
            '2FoFc_density_histogram': os.path.join(config.pdb_out_data_dir, 'checkmyblob', '2FoFc', 'density_hist'),
            '2FoFc_all_histogram': os.path.join(config.pdb_out_data_dir, 'checkmyblob', '2FoFc', 'all_hist'),
            '2FoFc_void_histogram': os.path.join(config.pdb_out_data_dir, 'checkmyblob', '2FoFc', 'void_hist'),
            'Fo_solvent_histogram': os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'Fo', 'solvent_hist'),
            'Fo_atom_histogram': os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'Fo', 'atom_hist'),
            'Fo_density_histogram': os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'Fo', 'density_hist'),
            'Fo_all_histogram': os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'Fo', 'all_hist'),
            'Fo_void_histogram': os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'Fo', 'void_hist'),
            'FoFc_histogram': os.path.join(config.pdb_out_data_dir, 'checkmyblob', 'diff_hist'),
        }

        self.filenames = {
            'cif': os.path.join(self.dir_static['cif'], '%s.cif' % pdb_code if cif_file is None else cif_file),
            'pdb': os.path.join(self.dir_static['pdb'], 'pdb%s.ent' % pdb_code if pdb_file is None else pdb_file),
            'mtz': os.path.join(self.dir_static['mtz'], '%s.mtz' % pdb_code if mtz_file is None else mtz_file),
            'map': os.path.join(self.dir_static['map'], '%s.map' % pdb_code),  # not used
            'edstats': os.path.join(self.dir_static['edstats'], '%s.out' % pdb_code),
            'result': os.path.join(self.directories['data'], '%s_results.txt' % pdb_code),
            'single_csv': os.path.join(self.directories['data'], '%s.csv' % pdb_code),
            'global_result': os.path.join(self.directories['data'], '%s_result_global.txt' % pdb_code),
            '2FoFc_solvent_histogram': os.path.join(self.directories['2FoFc_solvent_histogram'], '%s.png' % pdb_code),
            '2FoFc_atom_histogram': os.path.join(self.directories['2FoFc_atom_histogram'], '%s.png' % pdb_code),
            '2FoFc_density_histogram': os.path.join(self.directories['2FoFc_density_histogram'], '%s.png' % pdb_code),
            '2FoFc_all_histogram': os.path.join(self.directories['2FoFc_all_histogram'], '%s.png' % pdb_code),
            '2FoFc_void_histogram': os.path.join(self.directories['2FoFc_void_histogram'], '%s.png' % pdb_code),
            'Fo_solvent_histogram': os.path.join(self.directories['Fo_solvent_histogram'], '%s.png' % pdb_code),
            'Fo_atom_histogram': os.path.join(self.directories['Fo_atom_histogram'], '%s.png' % pdb_code),
            'Fo_density_histogram': os.path.join(self.directories['Fo_density_histogram'], '%s.png' % pdb_code),
            'Fo_all_histogram': os.path.join(self.directories['Fo_all_histogram'], '%s.png' % pdb_code),
            'Fo_void_histogram': os.path.join(self.directories['Fo_void_histogram'], '%s.png' % pdb_code),
            'FoFc_histogram': os.path.join(self.directories['FoFc_histogram'], '%s.png' % pdb_code),
            'ligand_dict': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dict', 'ligands.txt'),
        }

        self.temporary = {
            'map_fo': os.path.join(self.directories['temp'], '%s_%s_%s_%s_2fofc.map'),
            'mtz_refmac': os.path.join(self.directories['temp'], '%s_refmac.mtz' % pdb_code),
            'lib_cif_refmac': os.path.join(self.directories['temp'], '%s_lib_cif_refmac.cif' % pdb_code),
            'mtz_refmac_res': os.path.join(self.directories['temp'], '%s_refmac_res.map' % pdb_code),
            'cif_refmac': os.path.join(self.directories['temp'], '%s_refmac.cif' % pdb_code),
            'cif_cut': os.path.join(self.directories['temp'], '%s_cut.cif' % pdb_code),
            'cif': os.path.join(self.directories['temp'], '%s_refmac.cif' % pdb_code),
            'log_refmac': os.path.join(self.directories['temp'], '%s_refmac_log.txt' % pdb_code),
            'refmac_run_param_cache': os.path.join(self.directories['temp'], '%s_refmac_run_cache.txt' % pdb_code),
        }

        # Make sure the directories exists
        for path in self.directories.values():
            if not os.path.exists(path):
                os.makedirs(path)

        if overwrite:
            if os.path.exists(self.filenames['result']):
                os.remove(self.filenames['result'])
            if os.path.exists(self.filenames['global_result']):
                os.remove(self.filenames['global_result'])
            if os.path.exists(self.filenames['single_csv']):
                os.remove(self.filenames['single_csv'])

        # global results storage
        self.result_global_data = DataAggregator()

        self.result_global_data['pdb_code'] = pdb_code
        self.result_global_data['fo_col'] = self.options.MAP_F
        self.result_global_data['fc_col'] = self.options.MAP_PH
        self.result_global_data['weight_col'] = self.options.MAP_W

        self.result_global_data['grid_space'] = self.options.ORTH_GRID_SPACE
        self.result_global_data['solvent_radius'] = self.options.SOLVENT_RADIUS
        self.result_global_data['solvent_opening_radius'] = self.options.SOLVENT_OPENING_RADIUS
        self.LIGAND_DICT = get_ligand_atoms_dict(self.filenames['ligand_dict'])

        self.coefficients = ZernikeCoefficient.import_zernike_coefficient(
            os.path.join(os.path.dirname(__file__), "zernike_cc.txt"))

    def cleanup(self):
        # clean up
        for path in (self.directories['temp'],):
            if self.options.SAVE_CUT_MAP is False:
                self.logger.info('Removing... %s' % path)
                # shutil.rmtree(path)
            else:
                self.logger.info('Temporaty data not deleted' % path)

    def cut_non_standard(self, structure, cif_out, do_cut=True):
        if do_cut is True:
            res_list = []
            for res in structure.iter_fragments():
                res_list.append(res)

            for res in res_list:
                if res.res_name not in KEEP_RESIDUES:
                    structure.remove_fragment(res)

        # pyclipper.CifCut.exclude_res_from_mmcif(self.filenames['cif'], cif_out)
        FileIO.SaveStructure(fil=cif_out, struct=structure, format="CIF")
        return structure

    def run_refmac(self, LABIN_FP, LABIN_SIGFP):
        should_run = True

        CIF_CUT_MD5 = 'empty'
        with open(self.temporary['cif_cut'], 'rb') as cif_file:
            CIF_CUT_MD5 = hashlib.md5(cif_file.read()).hexdigest()

        MTZ_MD5 = 'empty'
        with open(self.filenames['mtz'], 'rb') as mtz_file:
            MTZ_MD5 = hashlib.md5(mtz_file.read()).hexdigest()

        options_dict = {
            'REFMAC_NCYC': self.options.REFMAC_NCYC if self.rerefine else 0,
            'REFMAC_WEIGHT': self.options.REFMAC_WEIGHT,
            'REFMAC_BFAC_SET': self.options.REFMAC_BFAC_SET,
            'CIF_CUT_MD5': CIF_CUT_MD5,
            'MTZ_MD5': MTZ_MD5,
        }

        if os.path.exists(self.temporary['refmac_run_param_cache']):
            with open(self.temporary['refmac_run_param_cache'], 'r') as cache_file:
                try:
                    cached_options = json.load(cache_file)
                except:
                    cached_options = {}

            if cached_options == options_dict and os.path.exists(self.temporary['mtz_refmac']) \
                    and CIF_CUT_MD5 != 'empty' and MTZ_MD5 != 'empty':
                should_run = False

        if should_run is False:
            return

        io = {
            'XYZIN': self.temporary['cif_cut'],
            'XYZOUT': self.temporary['cif_refmac'],
            'HKLIN': self.filenames['mtz'],
            'HKLOUT': self.temporary['mtz_refmac'],
            # 'LIBOUT': self.temporary['lib_cif_refmac'],
        }
        keywords = {
            'LABIN': 'FP=%s SIGFP=%s' % (LABIN_FP, LABIN_SIGFP),
            'LABOUT': 'FC=FC FWT=FWT PHIC=PHIC PHWT=PHWT DELFWT=DELFWT PHDELWT=PHDELWT FOM=FOM',
            'NCYC': self.options.REFMAC_NCYC if self.rerefine else 0,
            'WEIGHT': self.options.REFMAC_WEIGHT,
        }
        if self.options.REFMAC_BFAC_SET is not None and self.options.REFMAC_BFAC_SET > 0:
            keywords['NCYC'] = str(keywords['NCYC']) + ('\nBFAC SET %d' % self.options.REFMAC_BFAC_SET)

        self.logger.info('Running REFMAC')
        refmac = ccp4.CCP4_REFMAC(io, keywords)
        refmac.run()

        with open(self.temporary['log_refmac'], 'w') as refmac_log:
            refmac_log.write(refmac.output)

        with open(self.temporary['refmac_run_param_cache'], 'w') as cache_file:
            json.dump(options_dict, cache_file)

    def read_maps(self, max_resolution_limit):
        mtz_dir = self.temporary['mtz_refmac']

        # PHWT, PHIC, PHIC_ALL, PHIC_ALL_LS, PHDELWT,
        # FWT, FC, FC_ALL, FC_ALL_LS, FP, DELFWT
        # FOM

        if not os.path.exists(mtz_dir):
            self.logger.error("file %s does not exists" % mtz_dir)
            raise RefmacError("file %s does not exists" % mtz_dir)

        self.result_global_data['resolution_max_limit'] = max_resolution_limit
        if self.options.USE_TWO_FO_FC_MAP:
            self.two_Fo_minus_Fc = pyclipper.ClipperMap(mtz=mtz_dir, fo_col='FWT', fc_col='PHWT', weight_col='',
                                                        resolution_limit=max_resolution_limit,
                                                        target_grid_size=self.options.CLIPPER_GRID_SIZE_OTHER)
            self.two_Fo_minus_Fc_p1 = self.two_Fo_minus_Fc.get_values_p1()
            if self.options.MAP_F == 'FWT':
                self.result_global_data['resolution'] = self.two_Fo_minus_Fc.get_resolution()
                self.working_map = self.two_Fo_minus_Fc
                self.working_map_p1 = self.two_Fo_minus_Fc_p1

        if self.options.USE_FO_MAP:
            self.Fo = pyclipper.ClipperMap(mtz=mtz_dir, fo_col='FP', fc_col='PHIC_ALL', weight_col='',
                                           resolution_limit=max_resolution_limit,
                                           target_grid_size=self.options.CLIPPER_GRID_SIZE_OTHER)
            self.Fo_p1 = self.Fo.get_values_p1()
            if self.options.MAP_F == 'FP':
                self.result_global_data['resolution'] = self.Fo.get_resolution()
                self.working_map = self.Fo
                self.working_map_p1 = self.Fo

        self.Fo_minus_Fc = pyclipper.ClipperMap(mtz=mtz_dir, fo_col='DELFWT', fc_col='PHDELWT', weight_col='',
                                                resolution_limit=max_resolution_limit,
                                                target_grid_size=self.options.CLIPPER_GRID_SIZE_FOFC)
        if self.options.USE_FO_FC_MAP:
            self.Fo_minus_Fc_p1 = self.Fo_minus_Fc.get_values_p1()

        if self.options.MAP_F == 'DELFWT':
            self.result_global_data['resolution'] = self.Fo_minus_Fc.get_resolution()
            self.working_map = self.Fo_minus_Fc
            if self.options.USE_FO_FC_MAP:
                self.working_map_p1 = self.Fo_minus_Fc_p1

        if self.options.USE_FC_MAP:
            self.Fc = pyclipper.ClipperMap(mtz=mtz_dir, fo_col='FC_ALL', fc_col='PHIC_ALL', weight_col='',
                                           resolution_limit=max_resolution_limit,
                                           target_grid_size=self.options.CLIPPER_GRID_SIZE_OTHER)
            self.Fc_p1 = self.Fc.get_values_p1()

    def calculate_maps(self, max_resolution_limit, do_cut=True, fo='FP', sig_fo='SIGFP'):
        start = datetime.datetime.now()

        structure = self.get_structure()
        self.logger.info("Space Group %s" % structure.unit_cell.space_group.pdb_name)
        structure_cut = self.get_structure()
        structure_cut = self.cut_non_standard(structure_cut, self.temporary['cif_cut'], do_cut)

        self.run_refmac(fo, sig_fo)
        self.read_maps(max_resolution_limit)

        self.set_atoms(structure)
        self.set_atoms_cut(structure_cut)
        self.logger.info('calculate_maps took %s s' % (datetime.datetime.now() - start).total_seconds())

    def binary_opening(self, mask, size):
        # equivalent to
        # ball = morphology.ball(size, np.bool)
        # pad_mask = np.lib.pad(mask, size, 'wrap')
        # mask_opening_morphology = ndimage.morphology.binary_opening(pad_mask, ball)[size:-size, size:-size, size:-size]

        pad_mask = np.lib.pad(mask, size, 'wrap')
        dist_erosion = ndimage.distance_transform_edt(pad_mask == True)
        mask_envelope_erosion_dist = dist_erosion[size:-size, size:-size, size:-size] > size
        mask_envelope_erosion_dist = np.lib.pad(mask_envelope_erosion_dist, size, 'constant')
        dist_dilation = ndimage.distance_transform_edt(mask_envelope_erosion_dist == False)
        mask_opening_dist = (dist_dilation <= size)[size:-size, size:-size, size:-size]
        return mask_opening_dist

    def get_solvent_mask(self, working_map, atoms_position):
        start = datetime.datetime.now()
        dist_map = working_map.get_dist_map(atoms_position, self.options.SOLVENT_RADIUS)
        mask_frac_data = dist_map.get_values_p1()
        solvent_mask_and_void = mask_frac_data > self.options.SOLVENT_RADIUS

        size = int(self.options.SOLVENT_OPENING_RADIUS / self.options.ORTH_GRID_SPACE)

        self.solvent_mask = self.binary_opening(solvent_mask_and_void, size)

        self.void_mask = np.logical_and(solvent_mask_and_void, np.logical_not(self.solvent_mask))
        self.modeled_mask = ~solvent_mask_and_void

        self.result_global_data['solvent_mask_count'] = np.nansum(self.solvent_mask)
        self.result_global_data['void_mask_count'] = np.nansum(self.void_mask)
        self.result_global_data['modeled_mask_count'] = np.nansum(self.modeled_mask)
        self.result_global_data['solvent_ratio'] = float(self.result_global_data['solvent_mask_count']) / (
            self.result_global_data['solvent_mask_count'] + self.result_global_data['void_mask_count'] +
            self.result_global_data['modeled_mask_count']
        )
        self.logger.info('get_solvent_mask took %s s' % (datetime.datetime.now() - start).total_seconds())

    def calculate_map_stats(self):
        start = datetime.datetime.now()
        if self.options.USE_TWO_FO_FC_MAP:
            self.result_global_data['TwoFoFc_mean'] = np.nanmean(self.two_Fo_minus_Fc_p1)
            self.result_global_data['TwoFoFc_std'] = np.nanstd(self.two_Fo_minus_Fc_p1)
            self.result_global_data['TwoFoFc_square_std'] = np.nanstd(self.two_Fo_minus_Fc_p1 * self.two_Fo_minus_Fc_p1)
            self.result_global_data['TwoFoFc_min'] = np.nanmin(self.two_Fo_minus_Fc_p1)
            self.result_global_data['TwoFoFc_max'] = np.nanmax(self.two_Fo_minus_Fc_p1)

        if self.options.USE_FO_MAP:
            self.result_global_data['Fo_mean'] = np.nanmean(self.Fo_p1)
            self.result_global_data['Fo_std'] = np.nanstd(self.Fo_p1)
            self.result_global_data['Fo_square_std'] = np.nanstd(self.Fo_p1 * self.Fo_p1)
            self.result_global_data['Fo_min'] = np.nanmin(self.Fo_p1)
            self.result_global_data['Fo_max'] = np.nanmax(self.Fo_p1)

        if self.options.USE_FO_FC_MAP:
            self.result_global_data['FoFc_mean'] = np.nanmean(self.Fo_minus_Fc_p1)
            self.result_global_data['FoFc_std'] = np.nanstd(self.Fo_minus_Fc_p1)
            self.result_global_data['FoFc_square_std'] = np.nanstd(self.Fo_minus_Fc_p1 * self.Fo_minus_Fc_p1)
            self.result_global_data['FoFc_min'] = np.nanmin(self.Fo_minus_Fc_p1)
            self.result_global_data['FoFc_max'] = np.nanmax(self.Fo_minus_Fc_p1)
        else:
            mean, square_mean, cube_mean, variance, skewness, min, max = self.working_map.get_statistics()
            self.result_global_data['FoFc_mean'] = mean
            self.result_global_data['FoFc_std'] = math.sqrt(variance)
            self.result_global_data['FoFc_square_std'] = variance
            self.result_global_data['FoFc_min'] = min
            self.result_global_data['FoFc_max'] = max

        if self.options.USE_FC_MAP:
            self.result_global_data['Fc_mean'] = np.nanmean(self.Fc_p1)
            self.result_global_data['Fc_std'] = np.nanstd(self.Fc_p1)
            self.result_global_data['Fc_square_std'] = np.nanstd(self.Fc_p1 * self.Fc_p1)
            self.result_global_data['Fc_min'] = np.nanmin(self.Fc_p1)
            self.result_global_data['Fc_max'] = np.nanmax(self.Fc_p1)

        self.logger.info('calculate_map_stats took %s s' % (datetime.datetime.now() - start).total_seconds())

    def calculate_solvent_stats(self):
        start = datetime.datetime.now()
        solvent_mask_count = np.nansum(self.solvent_mask)

        maps = []
        prefixes = []
        if self.options.USE_FO_FC_MAP:
            maps.append(self.Fo_minus_Fc_p1)
            prefixes.append('FoFc')
        if self.options.USE_TWO_FO_FC_MAP:
            maps.append(self.two_Fo_minus_Fc_p1)
            prefixes.append('TwoFoFc')
        if self.options.USE_FO_MAP:
            maps.append(self.Fo_p1)
            prefixes.append('Fo')
        if self.options.USE_FC_MAP:
            maps.append(self.Fc_p1)
            prefixes.append('Fc')

        for density_map, prefix in zip(maps, prefixes):

            self.result_global_data['%s_bulk_mean' % prefix] = np.nan
            self.result_global_data['%s_bulk_std' % prefix] = np.nan
            if solvent_mask_count > 0:
                values = density_map[self.solvent_mask]
                self.result_global_data['%s_bulk_mean' % prefix] = np.nanmean(values)
                self.result_global_data['%s_bulk_std' % prefix] = np.nanstd(values)

            self.result_global_data['%s_void_mean' % prefix] = np.nan
            self.result_global_data['%s_void_std' % prefix] = np.nan
            if solvent_mask_count > 0:
                values = density_map[self.void_mask]
                self.result_global_data['%s_void_mean' % prefix] = np.nanmean(values)
                self.result_global_data['%s_void_std' % prefix] = np.nanstd(values)

            self.result_global_data['%s_modeled_mean' % prefix] = np.nan
            self.result_global_data['%s_modeled_std' % prefix] = np.nan
            if solvent_mask_count > 0:
                values = density_map[self.modeled_mask]
                self.result_global_data['%s_modeled_mean' % prefix] = np.nanmean(values)
                self.result_global_data['%s_modeled_std' % prefix] = np.nanstd(values)

        self.logger.info('calculate_solvent_stats took %s s' % (datetime.datetime.now() - start).total_seconds())

    def clean_maps(self):
        if self.options.USE_TWO_FO_FC_MAP:
            self.two_Fo_minus_Fc_p1 = None
            self.two_Fo_minus_Fc = None
        if self.options.USE_FO_MAP:
            self.Fo_p1 = None
            self.Fo = None
        if self.options.USE_FC_MAP:
            self.Fc_p1 = None
            self.Fc = None
        if self.options.USE_FO_FC_MAP:
            self.Fo_minus_Fc_p1 = None
            self.Fo_minus_Fc = None

        if self.options.SOLVENT_STATS:
            self.solvent_mask = None
            self.void_mask = None
            self.modeled_mask = None

    def read_edstats(self, filename):
        edstats = pd.read_csv(filename, sep=' ', header=0, skipinitialspace=True)
        return edstats

    def get_edstats_row(self, edstats, res, chain, res_number):
        row = None
        # print res, chain, res_number, edstats.dtypes['CI'], edstats.dtypes['RN'], type(edstats.dtypes['RN']), np.dtype(int), edstats.dtypes['RN'] is np.dtype(int)
        edstats['CI'] = edstats['CI'].astype(str)
        edstats['RT'] = edstats['RT'].astype(str)
        if edstats.dtypes['RN'] is np.dtype(int):
            row = edstats[(edstats['RT'] == str(res).strip()) & (edstats['CI'] == str(chain).strip()) & (
                edstats['RN'] == int(res_number))]
        else:
            row = edstats[(edstats['RT'] == str(res).strip()) & (edstats['CI'] == str(chain).strip()) & (
                edstats['RN'] == str(res_number))]
        try:
            row = row.iloc[0]
        except:
            edstats['RN'] = edstats['RN'].astype(str)
            for c in 'ABCDEF':
                edstats['RN'] = edstats['RN'].str.replace(c, '')
            # without_letters = edstats['RN'].str.extract('(\d)*(\w)?')
            # edstats['RN'] = without_letters[:, 0]
            # print edstats['RN']
            row = edstats[(edstats['RT'] == str(res).strip()) & (edstats['CI'] == str(chain).strip()) & (
                edstats['RN'] == str(res_number))]
            try:
                row = row.iloc[0]
            except:
                return None
        return row

    def add_edstats(self, edstats_row, result_data):
        if 'BAa' in edstats_row.index:
            tags_a = ['BAa', 'NPa', 'Ra', 'RGa', 'SRGa', 'CCSa', 'CCPa', 'ZOa', 'ZDa', 'ZD_minus_a', 'ZD_plus_a']
            tags_m = ['BAa', 'NPa', 'Ra', 'RGa', 'SRGa', 'CCSa', 'CCPa', 'ZOa', 'ZDa', 'ZD-a', 'ZD+a']
            for tag_a, tag_m in zip(tags_a, tags_m):
                result_data['local_%s' % tag_a] = edstats_row[tag_m] if edstats_row[tag_m] is not 'n/a' else np.nan
        else:
            tags_a = ['BAa', 'NPa', 'Ra', 'RGa', 'SRGa', 'CCSa', 'CCPa', 'ZOa', 'ZDa', 'ZD_minus_a', 'ZD_plus_a']
            tags_m = ['BAm', 'NPm', 'Rm', 'RGm', 'SRGm', 'CCSm', 'CCPm', 'ZOm', 'ZDm', 'ZD-m', 'ZD+m']
            for tag_a, tag_m in zip(tags_a, tags_m):
                result_data['local_%s' % tag_a] = edstats_row[tag_m] if edstats_row[tag_m] is not 'n/a' else np.nan

    def get_edstats(self, filename, res, result_data):
        row = None
        if os.path.isfile(filename):
            ff = open(filename, 'r')
            if len(ff.read().splitlines()) > 0:
                edstats = self.read_edstats(filename)
                row = self.get_edstats_row(edstats, res.res_name, res.chain_id, res.fragment_id)
            ff.close()
        if row is None:
            result_data['local_BAa'] = np.nan
            result_data['local_NPa'] = np.nan
            result_data['local_Ra'] = np.nan
            result_data['local_RGa'] = np.nan
            result_data['local_SRGa'] = np.nan
            result_data['local_CCSa'] = np.nan
            result_data['local_CCPa'] = np.nan
            result_data['local_ZOa'] = np.nan
            result_data['local_ZDa'] = np.nan
            result_data['local_ZD_minus_a'] = np.nan
            result_data['local_ZD_plus_a'] = np.nan
        else:
            self.add_edstats(row, result_data)

    def get_atoms_count(self, res, result_data):
        result_data['local_res_atom_count'] = np.nan
        result_data['local_res_atom_non_h_count'] = np.nan
        result_data['local_res_atom_non_h_occupancy_sum'] = np.nan
        result_data['local_res_atom_non_h_electron_sum'] = np.nan
        result_data['local_res_atom_non_h_electron_occupancy_sum'] = np.nan

        result_data['local_res_atom_C_count'] = np.nan
        result_data['local_res_atom_N_count'] = np.nan
        result_data['local_res_atom_O_count'] = np.nan
        result_data['local_res_atom_S_count'] = np.nan

        result_data['dict_atom_non_h_count'] = np.nan
        result_data['dict_atom_non_h_electron_sum'] = np.nan
        result_data['dict_atom_C_count'] = np.nan
        result_data['dict_atom_N_count'] = np.nan
        result_data['dict_atom_O_count'] = np.nan
        result_data['dict_atom_S_count'] = np.nan

        if res is None:
            return

        atom_non_h_count = 0
        atom_non_h_occupancy_sum = 0.0
        atom_non_h_electron_sum = 0.0
        atom_non_h_electron_occupancy_sum = 0.0

        atom_C_count = 0
        atom_N_count = 0
        atom_O_count = 0
        atom_S_count = 0

        for atom in res.iter_atoms():
            element = str(atom.element).strip().upper()
            if element != 'H':
                atom_non_h_count = atom_non_h_count + 1
                atom_non_h_occupancy_sum = atom_non_h_occupancy_sum + float(atom.occupancy)
                if element == 'X':
                    element = 'C'
                electrons = ELEMENTS_ELECTRONS[element]
                atom_non_h_electron_sum = atom_non_h_electron_sum + float(electrons)
                atom_non_h_electron_occupancy_sum = atom_non_h_electron_occupancy_sum + electrons * float(
                    atom.occupancy)
            if element == 'C':
                atom_C_count = atom_C_count + 1
            if element == 'N':
                atom_N_count = atom_N_count + 1
            if element == 'O':
                atom_O_count = atom_O_count + 1
            if element == 'S':
                atom_S_count = atom_S_count + 1

        result_data['local_res_atom_count'] = res.count_atoms()
        result_data['local_res_atom_non_h_count'] = atom_non_h_count
        result_data['local_res_atom_non_h_occupancy_sum'] = atom_non_h_occupancy_sum
        result_data['local_res_atom_non_h_electron_sum'] = atom_non_h_electron_sum
        result_data['local_res_atom_non_h_electron_occupancy_sum'] = atom_non_h_electron_occupancy_sum

        result_data['local_res_atom_C_count'] = atom_C_count
        result_data['local_res_atom_N_count'] = atom_N_count
        result_data['local_res_atom_O_count'] = atom_O_count
        result_data['local_res_atom_S_count'] = atom_S_count

        lig_name = res.res_name.strip().upper()
        if lig_name in self.LIGAND_DICT:
            lig_formula = self.LIGAND_DICT[lig_name][0]
            result_data['dict_atom_non_h_count'] = lig_formula.get_non_h_atoms_count()
            result_data['dict_atom_non_h_electron_sum'] = lig_formula.get_non_h_electron_count()
            result_data['dict_atom_C_count'] = lig_formula.get_element_count('C')
            result_data['dict_atom_N_count'] = lig_formula.get_element_count('N')
            result_data['dict_atom_O_count'] = lig_formula.get_element_count('O')
            result_data['dict_atom_S_count'] = lig_formula.get_element_count('S')

    def get_structure(self):
        # load structure
        st = FileIO.LoadStructure(file=self.filenames['pdb'], format='PDB')
        try:
            structure = FileIO.LoadStructure(file=self.filenames['cif'], format='CIF')
            self.logger.info("OPEN CIF %s" % self.filenames['cif'])
        except:
            structure = FileIO.LoadStructure(file=self.filenames['pdb'], format='PDB')
            self.logger.info("OPEN PDB %s" % self.filenames['pdb'])
        structure.unit_cell.space_group = st.unit_cell.space_group
        return structure

    def get_structure_refmac(self):
        # load structure
        return FileIO.LoadStructure(file=self.temporary['cif_refmac'])

    def set_atoms(self, struct):
        atoms_position = [atom.position for atom in struct.iter_atoms()]
        self.atoms_position = np.array(atoms_position, dtype=np.float)
        atoms_fragment_id = [atom.fragment_id for atom in struct.iter_atoms()]
        atoms_fragment_id = np.array(atoms_fragment_id)
        atoms_chain_id = [atom.chain_id for atom in struct.iter_atoms()]
        atoms_chain_id = np.array(atoms_chain_id)
        self.transformed_position = self.working_map.transform_all_rel_to_000(self.atoms_position)
        self.transformed_redundancy = self.transformed_position.shape[0] / self.atoms_position.shape[0]
        self.transformed_fragment_id = np.tile(atoms_fragment_id, self.transformed_redundancy)
        self.transformed_chain_id = np.tile(atoms_chain_id, self.transformed_redundancy)
        self.atoms_list = tuple((atom for atom in struct.iter_atoms()))

    def set_atoms_cut(self, struct):
        atoms_position = [atom.position for atom in struct.iter_atoms()]
        self.atoms_position_cut = np.array(atoms_position, dtype=np.float)
        atoms_fragment_id = [atom.fragment_id for atom in struct.iter_atoms()]
        atoms_fragment_id = np.array(atoms_fragment_id)
        atoms_chain_id = [atom.chain_id for atom in struct.iter_atoms()]
        atoms_chain_id = np.array(atoms_chain_id)
        self.transformed_position_cut = self.working_map.transform_all_rel_to_000(self.atoms_position_cut)
        self.transformed_redundancy_cut = self.transformed_position_cut.shape[0] / self.atoms_position_cut.shape[0]
        self.transformed_fragment_id_cut = np.tile(atoms_fragment_id, self.transformed_redundancy)
        self.transformed_chain_id_cut = np.tile(atoms_chain_id, self.transformed_redundancy)
        self.atoms_list_cut = tuple((atom for atom in struct.iter_atoms()))

    def get_min_max_atom_coord_in_frac(self, res, uc):
        min_xyz = uc.calc_orth_to_frac(res.iter_atoms().next().position)
        max_xyz = uc.calc_orth_to_frac(res.iter_atoms().next().position)

        for atom in res.iter_atoms():
            min_xyz = np.minimum(min_xyz, uc.calc_orth_to_frac(atom.position))
            max_xyz = np.maximum(max_xyz, uc.calc_orth_to_frac(atom.position))

        min_xyz = min_xyz - uc.calc_orth_to_frac([self.options.BORDER, self.options.BORDER, self.options.BORDER])
        max_xyz = max_xyz + uc.calc_orth_to_frac([self.options.BORDER, self.options.BORDER, self.options.BORDER])
        return min_xyz, max_xyz

    def get_min_max_atom_coord_in_orth(self, res):
        try:
            atom = res.iter_atoms().next()
        except StopIteration:
            return [-self.options.BORDER, -self.options.BORDER, -self.options.BORDER], [self.options.BORDER,
                                                                                        self.options.BORDER,
                                                                                        self.options.BORDER]
        else:
            min_xyz = atom.position
            max_xyz = atom.position
            for atom in res.iter_atoms():
                min_xyz = np.minimum(min_xyz, atom.position)
                max_xyz = np.maximum(max_xyz, atom.position)

        min_xyz = min_xyz - [self.options.BORDER, self.options.BORDER, self.options.BORDER]
        max_xyz = max_xyz + [self.options.BORDER, self.options.BORDER, self.options.BORDER]
        return min_xyz, max_xyz

    def get_min_max_uc_in_orth(self, uc):
        min_xyz = uc.calc_frac_to_orth(np.array((0.5, 0.5, 0.5)))
        max_xyz = uc.calc_frac_to_orth(np.array((0.5, 0.5, 0.5)))
        corners = [
            np.array((1.0, 0.0, 0.0)),
            np.array((1.0, 1.0, 0.0)),
            np.array((1.0, 0.0, 1.0)),
            np.array((1.0, 1.0, 1.0)),
            np.array((0.0, 0.0, 0.0)),
            np.array((0.0, 1.0, 0.0)),
            np.array((0.0, 0.0, 1.0)),
            np.array((0.0, 1.0, 1.0)),
        ]
        for position in corners:
            min_xyz = np.minimum(min_xyz, uc.calc_frac_to_orth(position))
            max_xyz = np.maximum(max_xyz, uc.calc_frac_to_orth(position))
        return min_xyz, max_xyz

    def cut_in_frac(self, res, struct, logger=None):
        min_xyz, max_xyz = self.get_min_max_atom_coord_in_frac(res, struct.unit_cell)
        if logger is not None and self.options.DEBUG_MORE is True:
            logger.info('MIN in FRAC %.3f %.3f %.3f' % tuple(min_xyz))
            logger.info('MAX in FRAC %.3f %.3f %.3f' % tuple(max_xyz))
        return self.working_map.cut_map(min_xyz, max_xyz)

    def do_cut_in_orth(self, min_xyz, max_xyz):
        x = np.arange(min_xyz[0], max_xyz[0] + self.options.ORTH_GRID_SPACE / 2.0, self.options.ORTH_GRID_SPACE)
        y = np.arange(min_xyz[1], max_xyz[1] + self.options.ORTH_GRID_SPACE / 2.0, self.options.ORTH_GRID_SPACE)
        z = np.arange(min_xyz[2], max_xyz[2] + self.options.ORTH_GRID_SPACE / 2.0, self.options.ORTH_GRID_SPACE)
        xx, yy, zz = np.meshgrid(x, y, z)
        points = np.vstack((xx, yy, zz)).reshape(3, -1).T
        orth_map = self.working_map.get_density_for_points(points)
        orth_map.shape = (len(y), len(x), len(z))
        orth_map = np.swapaxes(orth_map, 0, 1)
        return orth_map, min_xyz, max_xyz

    def cut_in_orth(self, res, logger=None):
        # returns evenly spaced np_array ir othogonal coordinates
        min_xyz, max_xyz = self.get_min_max_atom_coord_in_orth(res)
        if logger is not None and self.options.DEBUG_MORE is True:
            logger.info('MIN in ORTH %.3f %.3f %.3f' % tuple(min_xyz))
            logger.info('MAX in ORTH %.3f %.3f %.3f' % tuple(max_xyz))
        return self.do_cut_in_orth(min_xyz, max_xyz)

    def cut_uc_in_orth(self, uc):
        min_xyz, max_xyz = self.get_min_max_uc_in_orth(uc)
        return self.do_cut_in_orth(min_xyz, max_xyz)

    def mask_other_residues_get_atom(self, transformed_position, transformed_fragment_id, transformed_chain_id,
                                     atoms_list, min_xyz, max_xyz, res_fragment_id, res_chain_id):
        m = (transformed_position > (min_xyz - self.options.MASK_BORDER)) & (
            transformed_position < (max_xyz + self.options.MASK_BORDER))
        m = np.all(m, axis=1)

        m = m & (transformed_fragment_id != str(res_fragment_id))
        m = m & (transformed_chain_id != str(res_chain_id))
        transformed = transformed_position[m]
        atoms_idx = np.arange(m.shape[0])[m] % len(atoms_list)
        return transformed, atoms_idx

    def mask_other_residues(self, cut_map_orth, res_fragment_id, res_chain_id, transformed_position,
                            transformed_fragment_id, transformed_chain_id, atoms_list, min_xyz, max_xyz, diff_std,
                            logger=None, title=None, result_data=None):
        start = datetime.datetime.now()
        # get atoms for density masking
        transformed, atoms_idx = self.mask_other_residues_get_atom(
            transformed_position, transformed_fragment_id, transformed_chain_id, atoms_list, min_xyz, max_xyz,
            res_fragment_id, res_chain_id)

        if self.options.PRINT_3D_CUT_MASK:
            cut_map_orth2 = cut_map_orth.copy()

        # mask main chain and other atoms
        x = np.arange(min_xyz[0], max_xyz[0] + self.options.ORTH_GRID_SPACE / 2.0, self.options.ORTH_GRID_SPACE)
        y = np.arange(min_xyz[1], max_xyz[1] + self.options.ORTH_GRID_SPACE / 2.0, self.options.ORTH_GRID_SPACE)
        z = np.arange(min_xyz[2], max_xyz[2] + self.options.ORTH_GRID_SPACE / 2.0, self.options.ORTH_GRID_SPACE)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        mm = np.ones(xx.shape, dtype=bool)
        if logger is not None and self.options.DEBUG_MORE is True:
            logger.debug('SHAPE, %s %s %s, %s, %s' % (x.shape, y.shape, z.shape, xx.shape, yy.shape))
            logger.debug('    -> BOX %s %s' % (min_xyz - self.options.MASK_BORDER, max_xyz + self.options.MASK_BORDER))

        pad_size = int(math.floor(float(self.options.MASK_BORDER) / self.options.ORTH_GRID_SPACE))
        pad_dist = pad_size * self.options.ORTH_GRID_SPACE
        dist_map = cut_map_orth > self.get_threshold()
        dist_map = self.binary_dilation(dist_map, pad_size)

        for atom_idx, atom_xyz in izip(atoms_idx, transformed):
            atom = atoms_list[atom_idx]

            i, j, k = self.ort2grid(atom_xyz, min_xyz, max_xyz, self.options.ORTH_GRID_SPACE, pad_dist,
                                    shape=dist_map.shape)

            if atom.temp_factor * self.options.B_TO_U > 0 and atom.occupancy > 0:
                if dist_map[i][j][k] > 0:
                    loop_start = datetime.datetime.now()

                    element = atom.element if atom.element not in ('X',) else 'C'
                    atom_shape = pyclipper.ClipperAtomShape(
                        atom_xyz[0],
                        atom_xyz[1],
                        atom_xyz[2],
                        element,
                        atom.temp_factor * self.options.B_TO_U,
                        atom.occupancy
                    )
                    vrho = np.vectorize(atom_shape.rho)
                    atom_shape_rho = vrho(xx, yy, zz)
                    if logger is not None and self.options.DEBUG_MORE is True:
                        logger.debug('    -> INSIDE Res %s Id: %s %s' % (atom.res_name, atom.fragment_id, atom_xyz))
                        logger.debug('    .. element %s B fac %s occu %s' % (
                            element, atom.temp_factor * self.options.B_TO_U, atom.occupancy))
                        logger.debug('    -> arg min %s, arg max %s is nan count %s ' % (
                            np.nanmin(atom_shape_rho), np.nanmax(atom_shape_rho), np.sum(np.isnan(atom_shape_rho))))

                    # zwieksze nie RHO powoduje mniejsze wyciecie
                    mm = mm & (atom_shape_rho < self.options.RHO)
                    logger.debug('    ---- Res %s Id: %s (%s) ---- in envelope: %ss' % (
                        atom.res_name, atom.fragment_id, atom_xyz,
                        (datetime.datetime.now() - loop_start).total_seconds()))
                else:
                    logger.debug('    ---- Res %s Id: %s (%s) ---- not in envelope' % (
                        atom.res_name, atom.fragment_id, atom_xyz))

        if self.options.PRINT_3D_CUT_MASK:
            logger.debug(' MASKED MAP MIN MAX-> %s, %s' % (np.nanmin(cut_map_orth2[mm]), np.nanmax(cut_map_orth2[mm])))
            if title is None:
                title = "%s %s %s (%s)" % (
                    self.pdb_code, res_chain_id, res_fragment_id, result_data.get("res_coverage", ""))

            cut_map_orth2[~mm] = 0.0
            if self.options.DEBUG_MORE_TITLE_CONTAINS in title:
                print_3d_mesh(cut_map_orth2, self.get_threshold(), 0.1, ~mm, 0.5, 0.1,
                              title=("MAP AFTER ATOM MASK %s" % title), logger=self.logger,
                              grid_space=self.options.ORTH_GRID_SPACE)

        mm = ~mm
        blob_mask = cut_map_orth > (self.get_threshold())
        cut_volume = np.nansum(blob_mask & mm) * (self.options.ORTH_GRID_SPACE ** 3)
        result_data['local_cut_by_mainchain_volume'] = cut_volume
        cut_map_orth[mm] = 0.0

        logger.debug('    ---- envelope ---- mask %s cut_volume %s' % (np.nansum(mm), cut_volume))
        logger.debug('mask_other_residues took %s s' % ((datetime.datetime.now() - start).total_seconds()))

        self.get_nearby_atom_count(cut_map_orth > 0, min_xyz, max_xyz, atoms_list, atoms_idx, transformed, result_data,
                                   logger)

        return cut_map_orth

    def binary_dilation(self, mask, size, subtract_mask=True, cut_pad=False):
        # equivalent of, however binary_dilation is slower
        # mask = np.lib.pad(mask, size, 'constant')
        # ball = morphology.ball(size, np.bool)
        # mask_envelope = ndimage.morphology.binary_dilation(mask, ball)

        pad_mask = np.lib.pad(mask, size, 'constant')
        dist = ndimage.distance_transform_edt(pad_mask == 0)
        mask_dilation = dist <= size

        if subtract_mask is True:
            mask_dilation[pad_mask > 0] = 0

        if cut_pad is True:
            mask_dilation = mask_dilation[size:-size, size:-size, size:-size]

        return mask_dilation

    def get_nearby_atom_count(self, mask, min_xyz, max_xyz, atoms_list, atoms_idx, transformed, result_data, logger):
        start = datetime.datetime.now()

        count_C = 0
        count_O = 0
        count_S = 0
        count_N = 0
        count_other = 0

        pad_size = int(math.floor(float(self.options.NEAR_ATOM_DIST) / self.options.ORTH_GRID_SPACE))
        pad_dist = pad_size * self.options.ORTH_GRID_SPACE
        mask_envelope = self.binary_dilation(mask, pad_size)

        if logger is not None:
            logger.debug(
                'min %s max %s min+pad %s max+pad %s' % (min_xyz, max_xyz, min_xyz - pad_dist, max_xyz + pad_dist))

        for atom_idx, atom_xyz in izip(atoms_idx, transformed):
            atom = atoms_list[atom_idx]
            element = atom.element if atom.element not in ('X',) else 'C'
            element = element.strip().upper()

            if element != 'H':
                x = atom_xyz[0]
                y = atom_xyz[1]
                z = atom_xyz[2]
                i = int(math.floor((x - (min_xyz[0] - pad_dist)) / self.options.ORTH_GRID_SPACE))
                j = int(math.floor((y - (min_xyz[1] - pad_dist)) / self.options.ORTH_GRID_SPACE))
                k = int(math.floor((z - (min_xyz[2] - pad_dist)) / self.options.ORTH_GRID_SPACE))

                if mask_envelope[i][j][k] > 0:
                    if logger is not None:
                        logger.info('    %s %s %s %s' % (atom.element, atom.chain_id, atom.fragment_id, atom_xyz))
                    if element == 'C':
                        count_C += 1
                    elif element == 'O':
                        count_O += 1
                    elif element == 'N':
                        count_N += 1
                    elif element == 'S' or element == 'SE':
                        count_S += 1
                    else:
                        count_other += 1

        count = {'C': count_C, 'O': count_O, 'N': count_N, 'S': count_S, 'other': count_other}
        if logger is not None:
            logger.info('%s' % count)
        if result_data is not None:
            for element, value in count.items():
                result_data['local_near_cut_count_%s' % element] = value
        if logger is not None:
            logger.info('get_nearby_atom_count took %s s' % ((datetime.datetime.now() - start).total_seconds()))
        return count

    def count_segments(self, map, mask):
        distance = ndimage.distance_transform_edt(map)
        distance_maxi = feature.peak_local_max(distance, indices=False, exclude_border=False,
                                               min_distance=self.options.PEAK_MAX_DISTACNE, labels=mask)
        if distance_maxi.sum() > 0:
            markers = morphology.label(distance_maxi)
            markers[~mask] = -1
            return markers[markers > 0].sum()
        return 0

    def calc_decey(self, cut_map_orth, result_data):
        self.result_global_data['part_step_FoFc_std_min'] = self.options.DENSITY_STD_THRESHOLD
        self.result_global_data[
            'part_step_FoFc_std_max'] = self.options.DENSITY_STD_THRESHOLD + self.options.PART_STEP_SIZE * (
            self.options.PART_STEP_COUNT + 0.5)
        self.result_global_data['part_step_FoFc_std_step'] = self.options.PART_STEP_SIZE
        step_min = self.result_global_data['FoFc_std'] * self.result_global_data['part_step_FoFc_std_min']
        step_max = self.result_global_data['FoFc_std'] * self.result_global_data['part_step_FoFc_std_max']
        step_step = self.result_global_data['FoFc_std'] * self.result_global_data['part_step_FoFc_std_step']

        for ii, density_threshold in enumerate(drange(step_min, step_max, step_step)):
            mask = cut_map_orth > density_threshold
            cut_map_orth_mask = cut_map_orth * mask

            part_cut_map_orth, part_mask, part_parts_count = self.cut_small_parts(cut_map_orth_mask, density_threshold)

            result_data['part_%02d_shape_segments_count' % ii] = 0
            result_data['part_%02d_density_segments_count' % ii] = 0
            if mask.sum() > 0:
                result_data['part_%02d_shape_segments_count' % ii] = self.count_segments(part_mask, part_mask)
                result_data['part_%02d_density_segments_count' % ii] = self.count_segments(part_cut_map_orth, part_mask)

            self.calc_stats(part_cut_map_orth, part_mask, part_parts_count, result_data, 'part_%02d' % ii)
            self.calc_moment_invariant(part_cut_map_orth, part_mask, result_data, 'part_%02d' % ii)

    def calc_moment_invariant(self, cut_map_orth, mask, result_data, result_prefix='local'):
        label_mask = mask.astype(int)

        keys = [
            'O3', 'O4', 'O5', 'FL',
            'O3_norm', 'O4_norm', 'O5_norm', 'FL_norm',
            'I1', 'I2', 'I3', 'I4', 'I5', 'I6',
            'I1_norm', 'I2_norm', 'I3_norm', 'I4_norm', 'I5_norm', 'I6_norm',
            'M000',
            'CI',
        ]
        keys_eigen = ['E3_E1', 'E2_E1', 'E3_E2']
        keys_eigen_sqrt = ['E1', 'E2', 'E3']

        moments_cache = MomentCache(label_mask, normalize=False)
        invariants = GeometricalInvariantCache(moments_cache)

        for key in keys:
            result_data['%s_shape_%s' % (result_prefix, key)] = invariants.invariants[key]
        for key in keys_eigen:
            result_data['%s_shape_%s' % (result_prefix, key)] = np.abs(invariants.invariants[key]) if not np.isnan(
                invariants.invariants[key]) else invariants.invariants[key]
        for key in keys_eigen_sqrt:
            result_data['%s_shape_sqrt_%s' % (result_prefix, key)] = np.sqrt(
                np.abs(invariants.invariants[key])) if not np.isnan(invariants.invariants[key]) else \
                invariants.invariants[key]

        moments_cache = MomentCache(label_mask * cut_map_orth, normalize=False)
        invariants = GeometricalInvariantCache(moments_cache)

        for key in keys:
            result_data['%s_density_%s' % (result_prefix, key)] = invariants.invariants[key]
        for key in keys_eigen:
            result_data['%s_density_%s' % (result_prefix, key)] = np.abs(invariants.invariants[key]) if not np.isnan(
                invariants.invariants[key]) else invariants.invariants[key]
        for key in keys_eigen_sqrt:
            result_data['%s_density_sqrt_%s' % (result_prefix, key)] = np.sqrt(
                np.abs(invariants.invariants[key])) if not np.isnan(invariants.invariants[key]) else \
                invariants.invariants[key]

        moments = MomentCache(label_mask, normalize=True)
        zernike_moment = ZernikeMomentCache(self.coefficients, moments, order=7)
        for key, val in zernike_moment.invariants.iteritems():
            result_data['%s_shape_Z_%s_%s' % (result_prefix, key[0], key[1])] = val

        moments = MomentCache(label_mask * cut_map_orth, normalize=True)
        zernike_moment = ZernikeMomentCache(self.coefficients, moments, order=7)
        for key, val in zernike_moment.invariants.iteritems():
            result_data['%s_density_Z_%s_%s' % (result_prefix, key[0], key[1])] = val

    def calc_stats(self, cut_map_orth, mask, count_parts, result_data, result_prefix='local'):
        volume = 0.0
        electrons = 0.0
        mean = 0.0
        std = 0.0
        mmin = 0.0
        mmax = 0.0
        skewness = 0.0
        grid_volume = self.options.ORTH_GRID_SPACE ** 3

        cut_map_orth_masked = cut_map_orth[mask > 0]
        if cut_map_orth_masked.flatten().shape[0] > 0:
            volume = np.nansum(mask > 0) * grid_volume
            electrons = np.nansum(cut_map_orth_masked) * grid_volume
            mean = np.nanmean(cut_map_orth_masked)
            std = np.nanstd(cut_map_orth_masked)
            if 'part' not in result_prefix:
                mmin = np.nanmin(cut_map_orth_masked)
            mmax = np.nanmax(cut_map_orth_masked)

            map_minus_mean_tmp = cut_map_orth_masked - mean
            np.power(map_minus_mean_tmp, 3, map_minus_mean_tmp)
            skewness = np.power(np.nanmean(map_minus_mean_tmp), 1 / 3.0)
            del map_minus_mean_tmp

        result_data['%s_volume' % result_prefix] = volume
        result_data['%s_electrons' % result_prefix] = electrons
        result_data['%s_mean' % result_prefix] = mean
        result_data['%s_std' % result_prefix] = std
        if 'part' not in result_prefix:
            result_data['%s_min' % result_prefix] = mmin
        result_data['%s_max' % result_prefix] = mmax
        result_data['%s_max_over_std' % result_prefix] = mmax / self.result_global_data['FoFc_std']
        result_data['%s_skewness' % result_prefix] = skewness
        if 'local' not in result_prefix:
            result_data['%s_parts' % result_prefix] = count_parts

    def cut_small_parts(self, cut_map_orth, threshold):
        mask = cut_map_orth > threshold
        cut_map_orth_mask = cut_map_orth * mask

        grid_volume = self.options.ORTH_GRID_SPACE ** 3
        label_im, nb_labels = ndimage.label(mask, structure=np.ones((3, 3, 3)))
        blob_volume = ndimage.sum(mask, label_im, range(nb_labels + 1)) * grid_volume
        blob_electrons_max = ndimage.maximum(cut_map_orth_mask, label_im, range(nb_labels + 1))
        cut_map_orth_mask = None
        del cut_map_orth_mask

        mask_big_parts = np.zeros_like(cut_map_orth, dtype=int)
        self.logger.info("===== cut_small_parts ==== threshold: %s" % threshold)
        parts = 0
        for lab, volume in enumerate(blob_volume):
            if volume > self.options.MIN_VOLUME_LIMIT and blob_electrons_max[lab] > threshold:
                self.logger.info('    BIG Label: %s (threshold: %s) volume: %.3f, max: %.3f' % (
                    lab, threshold, volume, blob_electrons_max[lab]))
                mask_big_parts[label_im == lab] = 1
                parts += 1
            else:
                self.logger.info("masking small part %s (threshold: %s, min volume: %s) of volume %s and max %s" % (
                    lab, threshold, self.options.MIN_VOLUME_LIMIT, volume, blob_electrons_max[lab]))

        return cut_map_orth * mask_big_parts, mask_big_parts, parts

    def process_blob(self, cut_map_orth, mask, parts_count, title, result_data):
        if self.options.PRINT_3D_MESH:
            if self.options.DEBUG_MORE_TITLE_CONTAINS in title:
                print_3d_mesh(cut_map_orth, self.get_threshold(), title=("BLOB CLEAN %s" % title),
                              logger=self.logger, grid_space=self.options.ORTH_GRID_SPACE)

        if self.options.SHOW_DECAY is True:
            self.calc_decey(cut_map_orth, result_data)

        for key, val in self.result_global_data.iteritems():
            result_data[key] = val
        result_data.save_results(self.filenames['result'])

    def analyze_res(self):
        struct = self.get_structure()
        res_done_count = 0
        # for res in struct.iter_non_standard_residues():
        for res in struct.iter_fragments():
            if res.res_name not in IGNORED_RESIDUES and res.count_atoms() > 0:
                res_done_count = res_done_count + 1
                result_data = DataAggregator()
                title = "%s %s %s %s" % (str(self.pdb_code), str(res.res_name), str(res.fragment_id), str(res.chain_id))

                result_data['title'] = title
                result_data['pdb_code'] = self.pdb_code
                result_data['res_name'] = res.res_name
                result_data['res_id'] = res.fragment_id
                result_data['chain_id'] = res.chain_id
                if self.options.EDSTATS:
                    self.get_edstats(self.filenames['edstats'], res, result_data)
                self.get_atoms_count(res, result_data)

                if self.options.DEBUG_MORE is True:
                    for atom in res.iter_atoms():
                        self.logger.info("%s, %s" % (title, str(atom.position)))

                # save the part of the that was cat
                if self.options.SAVE_CUT_MAP is True:
                    tmp_map_fo = self.temporary['map_fo'] % (self.pdb_code, res.chain_id, res.fragment_id, res.res_name)
                    cut_map = self.cut_in_frac(res, struct, self.logger)
                    cut_map.save(tmp_map_fo)

                # get numpy array in orth
                cut_map_orth, min_xyz, max_xyz = self.cut_in_orth(res, self.logger)

                # get atoms for density masking
                cut_map_orth = self.mask_other_residues(
                    cut_map_orth,
                    res.fragment_id,
                    res.chain_id,
                    self.transformed_position,
                    self.transformed_fragment_id,
                    self.transformed_chain_id,
                    self.atoms_list,
                    min_xyz, max_xyz,
                    self.result_global_data['FoFc_std'],
                    self.logger,
                    result_data=result_data,
                )

                cut_map_orth, mask, parts_count = self.cut_small_parts(cut_map_orth, self.get_threshold())

                self.process_blob(cut_map_orth, parts_count, title, result_data)

        if res_done_count == 0:
            result_data = DataAggregator()
            result_data.save_results(self.filenames['result'])
        self.result_global_data.save_results(self.filenames['global_result'])

    def get_blob_min_max_in_orth(self, blob):
        min_box = np.array(blob.min_box_o) - [self.options.BORDER, self.options.BORDER, self.options.BORDER]
        max_box = np.array(blob.max_box_o) + [self.options.BORDER, self.options.BORDER, self.options.BORDER]
        return min_box, max_box

    def get_blob_min_max_in_frac(self, blob):
        return np.array(blob.min_box_f), np.array(blob.max_box_f)

    def get_blobs(self, water_wolume, threshold, merge_distance):
        map_clusters = self.working_map.get_mask_map(threshold, water_wolume, custom_skeleton=True, verbose=False)
        self.logger.debug('Blobs after get_mask_map:', len(map_clusters))
        out_blobs = []
        for i_cluster, cluster in enumerate(map_clusters):
            self.logger.debug(i_cluster, cluster.volume, cluster.density, 'loc_m', len(cluster.map_local_max_o), 'skel',
                              len(cluster.skeleton))
            blob = Blob(
                volume=cluster.volume,
                density=cluster.density,
                min_box_o=cluster.min_box_o,
                max_box_o=cluster.max_box_o,
                max_point_box_o_list=[cluster.max_point_box_o],
                local_maxi_o=cluster.map_local_max_o,
                skeleton=cluster.skeleton,
                surface=cluster.surface,
                children=None,
                map=self.working_map,
            )

            out_blobs.append(blob)

        out_blobs_merged = Blob.merge(out_blobs, merge_distance)
        self.logger.debug('Blobs after merge:', len(out_blobs_merged))
        out_blobs_merged_with_volume = [blob for blob in out_blobs_merged if
                                        blob.volume > self.options.MIN_VOLUME_LIMIT_MERGED_BLOB]
        self.logger.debug('Blobs after merge and volue limit (%s): %s' % (
            self.options.MIN_VOLUME_LIMIT_MERGED_BLOB, len(out_blobs_merged_with_volume)))

        if self.options.DEBUG_MORE:
            for bb in out_blobs_merged_with_volume:
                self.logger.debug("Volume: %s\nDensity: %s\nGlobal max: %s\nLocal max: %s\n" % (
                    bb.volume, bb.density, bb.max_point_box_o_list, bb.local_maxi_o))
                if bb.children is not None:
                    for bbc in bb.children:
                        self.logger.debug(
                            "    Volume: %s\n    Density: %s\n    Global max: %s\n    Local max: %s\n    Skeleton len %s\n" % (
                                bbc.volume, bbc.density, bbc.max_point_box_o_list, bbc.local_maxi_o, len(bbc.skeleton)))

        return out_blobs_merged_with_volume

    def mask_other_parts(self, cut_map_orth, min_xyz_o, max_xyz_o, selection_points_o, density_threshold, title,
                         blob=None):
        self.logger.debug('MIN BOX ORTH %s' % min_xyz_o)
        self.logger.debug('MAX BOX ORTH %s' % max_xyz_o)
        self.logger.debug('SELECTION POINTS ORTH %s' % selection_points_o)
        mask = cut_map_orth > density_threshold
        self.logger.info('MASK FIRST (sum) %s' % np.sum(mask))
        label_im, nb_labels = ndimage.label(mask, structure=np.ones((3, 3, 3)))

        blob_labels = set()

        for blob_max_point_box_o in selection_points_o:
            i, j, k = self.ort2grid(blob_max_point_box_o, min_xyz_o, max_xyz_o, self.options.ORTH_GRID_SPACE,
                                    shape=cut_map_orth.shape)
            label = label_im[i, j, k]

            if label == 0:
                label_fix = {
                    label_im[i + 1, j, k], label_im[i - 1, j, k],
                    label_im[i, j - 1, k], label_im[i, j + 1, k],
                    label_im[i, j, k - 1], label_im[i, j, k + 1],
                    label_im[i + 1, j + 1, k], label_im[i - 1, j - 1, k],
                    label_im[i + 1, j, k + 1], label_im[i - 1, j, k - 1],
                    label_im[i, j + 1, k + 1], label_im[i, j - 1, k - 1],
                    label_im[i + 1, j - 1, k], label_im[i - 1, j + 1, k],
                    label_im[i, j + 1, k - 1], label_im[i, j - 1, k + 1],
                    label_im[i + 1, j, k - 1], label_im[i - 1, j, k + 1],
                }
                label_fix.discard(0)

                if len(label_fix) == 1:
                    label = label_fix.pop()
                    self.logger.info("fixed ZERO label")
                else:
                    self.logger.info("ambiguous ZERO label")

            if label == 0:
                self.logger.info("ZERO label seems to be wrong")

                val = cut_map_orth[i, j, k]
                mask_val = mask[i, j, k]
                label_val = label_im[i, j, k]
                self.logger.info("density_threshold: %s" % density_threshold)
                self.logger.info(
                    "ii=%s jj=%s kk=%s val=%s mask_val=%s label_val=%s" % (i, j, k, val, mask_val, label_val))

                for ii in range(-1, 2, 1):
                    for jj in range(-1, 2, 1):
                        for kk in range(-1, 2, 1):
                            val = cut_map_orth[i + ii, j + jj, k + kk]
                            mask_val = mask[i + ii, j + jj, k + kk]
                            label_val = label_im[i + ii, j + jj, k + kk]
                            self.logger.info("ii=%s jj=%s kk=%s val=%s mask_val=%s label_val=%s" % (
                                ii, jj, kk, val, mask_val, label_val))

                if False and self.options.PRINT_3D_CUT_MASK:
                    if self.options.DEBUG_MORE_TITLE_CONTAINS in title:
                        print_3d_mesh(
                            cut_map_orth,
                            density_threshold,
                            blob=blob,
                            map_min_o=min_xyz_o,
                            special_points=[blob_max_point_box_o],
                            title=('LABEL 0; density %s; %s' % (density_threshold, title)),
                            logger=self.logger,
                            grid_space=self.options.ORTH_GRID_SPACE,
                        )
            else:
                blob_labels.add(label)

            self.logger.debug('LABEL %s' % label)

        mask[~(np.in1d(label_im, list(blob_labels)).reshape(mask.shape))] = False
        self.logger.info('MASK SECOND (sum) %s' % np.sum(mask))
        cut_map_orth[~mask] = 0.0

        if self.options.PRINT_3D_CUT_MASK:
            if self.options.DEBUG_MORE_TITLE_CONTAINS in title:
                print_3d_mesh(
                    cut_map_orth,
                    density_threshold,
                    blob=blob,
                    map_min_o=min_xyz_o,
                    special_points=blob.skeleton,
                    title=(
                        'MAP cut other parts; density %s; %s; parts %s' % (
                            density_threshold, title, len(blob.children))),
                    logger=self.logger,
                    grid_space=self.options.ORTH_GRID_SPACE,
                    morphology_skel3d=True,
                )

        return cut_map_orth

    def get_closest_res_name(self, max_point_in_orth):
        max_point_in_orth = np.array(max_point_in_orth)
        # get atoms for density masking
        m = (self.transformed_position > (max_point_in_orth - 2.0 * self.options.MASK_BORDER)) & (
            self.transformed_position < (max_point_in_orth + 2.0 * self.options.MASK_BORDER))
        m = np.all(m, axis=1)
        transformed_position = self.transformed_position[m]

        atoms_idx = np.arange(m.shape[0])[m] % len(self.atoms_list)

        res_name = 'BLOB'
        chain_id = None
        fragment_id = None
        atom = None

        min_idx, min_dist = self.working_map.get_closest_idx(np.asfarray(transformed_position, dtype='float'),
                                                             atoms_idx, np.asfarray(max_point_in_orth, dtype='float'))

        if min_idx is not None:
            atom = self.atoms_list[min_idx]
            fragment_id = atom.fragment_id
            chain_id = atom.chain_id
            res_name = atom.res_name
        else:
            min_dist = 9999999.0
        return chain_id, res_name, fragment_id, min_dist, atom

    def ort2grid(self, xyz, min_orth, max_orth, grid_space, pad=0.0, shape=None):
        i = int(math.floor((xyz[0] - (min_orth[0] - pad)) / grid_space))
        j = int(math.floor((xyz[1] - (min_orth[1] - pad)) / grid_space))
        k = int(math.floor((xyz[2] - (min_orth[2] - pad)) / grid_space))

        if shape is not None:
            if i < 0 or i >= shape[0]:
                self.logger.error(
                    "!!! ort2grid returned i outside of shape: i=%d, shape=%d xyz=%s min=%s max=%s pad=%s", i, shape[0],
                    xyz, min_orth, max_orth, pad)
                raise Exception("ort2grid returned i outside of shape")
            if j < 0 or j >= shape[1]:
                self.logger.error(
                    "!!! ort2grid returned j outside of shape: j=%d, shape=%d xyz=%s min=%s max=%s pad=%s", j, shape[1],
                    xyz, min_orth, max_orth, pad)
                raise Exception("ort2grid returned j outside of shape")
            if k < 0 or k >= shape[2]:
                self.logger.error(
                    "!!! ort2grid returned k outside of shape: k=%d, shape=%d xyz=%s min=%s max=%s pad=%s", k, shape[2],
                    xyz, min_orth, max_orth, pad)
                raise Exception("ort2grid returned k outside of shape")
        return (i, j, k)

    def find_atoms(self, min_xyz, max_xyz):
        m = (self.transformed_position > (min_xyz)) & (self.transformed_position < (max_xyz))
        m = np.all(m, axis=1)
        transformed_position = self.transformed_position[m]
        # moze unikalne indeksy tylko?
        atoms_idx = np.arange(m.shape[0])[m] % len(self.atoms_list)
        # dziele przez liczbe atomow i liczbe shiftow
        atoms_symetry_idx = np.arange(m.shape[0])[m] // (len(self.atoms_list) * 27)
        return transformed_position, atoms_idx, atoms_symetry_idx

    def format_probabilities(self, result_data, res_probablities, key, minimal_probablility=2.0, frag_probablities=None,
                             key_frag=None):
        # clean parts with probability < minimal_probablility
        for code in list(res_probablities.keys()):
            res_probablities[code] = round(res_probablities[code], 1)
            if frag_probablities is not None:
                frag_probablities[code] = round(frag_probablities[code], 1)

            if res_probablities[code] < minimal_probablility:
                del res_probablities[code]
                if frag_probablities is not None:
                    del frag_probablities[code]

        if res_probablities:
            result_data[key] = "'%s'" % json.dumps(res_probablities)
        else:
            result_data[key] = np.nan

        if frag_probablities is not None:
            result_data[key_frag] = "'%s'" % json.dumps(frag_probablities)
        else:
            result_data[key_frag] = np.nan

    def get_most_probable_res_name(self, cut_map_orth, density_threshold, min_xyz_out, max_xyz_out, result_data):
        start = datetime.datetime.now()

        transformed_position, atoms_idx, atoms_symetry_idx = self.find_atoms(min_xyz_out, max_xyz_out)

        atom_codes = {}
        markers = np.zeros(cut_map_orth.shape)
        for iatom, atom_xyz in enumerate(transformed_position):
            i, j, k = self.ort2grid(atom_xyz, min_xyz_out, max_xyz_out, self.options.ORTH_GRID_SPACE,
                                    shape=cut_map_orth.shape)

            # print 'xyz', atom.res_name, atom.chain_id, atom.fragment_id, atom_xyz, x,y,z, markers.shape
            # print 'marker', atoms_idx[iatom], atoms_symetry_idx[iatom]

            # tu mozna zamiast iatom+1 brac atoms_idx[iatom]+1
            # atom = self.atoms_list[atom_selector].fragment_id
            markers[i, j, k] = iatom + 1
            atom_codes[iatom + 1] = (atoms_idx[iatom], atoms_symetry_idx[iatom])

        mask = cut_map_orth > density_threshold
        all_counts = np.nansum(mask)
        dist = -ndimage.distance_transform_edt(markers == 0)
        try:
            codes = segmentation.random_walker(dist, markers, mode='cg_mg', tol=0.01, beta=100)
            codes[~mask] = -1
        except ValueError:
            self.logger.error('ValueError during random_walker segmentation')
            codes = -np.ones(cut_map_orth.shape)

        unique, counts = np.unique(codes, return_counts=True)

        del mask

        res_probablities = {}
        res_atom = {}
        for x, x_count in zip(unique, counts):
            if x > 0:
                prob = 100.0 * x_count / all_counts

                atom_selector = atom_codes[x][0]
                atom = self.atoms_list[atom_selector]

                transformation_selector = atom_codes[x][1]

                # transformacje dziwnie dzialaja
                code = "%s:%s:%s:%s" % (atom.res_name, atom.chain_id, atom.fragment_id, transformation_selector)

                res_probablities[code] = res_probablities.get(code, 0.0) + prob
                res_atom[code] = atom

                print x, x_count, prob, atom_codes[x]

        chain_id = None
        res_name = None
        fragment_id = None
        res = None
        max_probability = -1

        for code, code_prob in sorted(res_probablities.items(), key=operator.itemgetter(1), reverse=True):
            print 'rw->', code, code_prob
            if max_probability < code_prob and code_prob > self.options.MINIMAL_SEGMENT_PROBABILITY:
                max_probability = code_prob
                fragment_id = res_atom[code].fragment_id
                chain_id = res_atom[code].chain_id
                res_name = res_atom[code].res_name
                res = res_atom[code].get_fragment()

        self.format_probabilities(result_data, res_probablities, "res_walker_prob",
                                  self.options.MINIMAL_SEGMENT_PROBABILITY)

        self.logger.debug('get_most_probable_res_name took %s s' % ((datetime.datetime.now() - start).total_seconds()))
        return chain_id, res_name, fragment_id, max_probability, res

    def get_res_name_with_spheres(self, cut_map_orth, density_threshold, min_xyz_out, max_xyz_out, result_data,
                                  max_radii):
        max_radii_in_pixels = max_radii / self.options.ORTH_GRID_SPACE
        start = datetime.datetime.now()

        transformed_position, atoms_idx, atoms_symetry_idx = self.find_atoms(min_xyz_out, max_xyz_out)

        fragment_xyz = {}
        fragment_atom = {}

        for iatom, atom_xyz in enumerate(transformed_position):
            i, j, k = self.ort2grid(atom_xyz, min_xyz_out, max_xyz_out, self.options.ORTH_GRID_SPACE,
                                    shape=cut_map_orth.shape)
            atom = self.atoms_list[atoms_idx[iatom]]

            self.logger.debug(
                "ijk=({},{},{}), xyz=({},{},{}), atom_xyz=({},{},{}), chain={}, res={}, fragment={}, element={}".format(
                    i, j, k, atom_xyz[0], atom_xyz[1], atom_xyz[2], atom.position[0], atom.position[1],
                    atom.position[2],
                    atom.chain_id, atom.res_name, atom.fragment_id, atom.element
                ))

            code = "%s:%s:%s:%s" % (atom.res_name, atom.chain_id, atom.fragment_id, atoms_symetry_idx[iatom])
            fragment_xyz.setdefault(code, [(i, j, k)]).append((i, j, k))
            fragment_atom[code] = atom

        mask = cut_map_orth > density_threshold
        all_counts = np.nansum(mask)

        chain_id = None
        res_name = None
        fragment_id = None
        res = None
        max_probability = 0
        second_max_probability = 0
        max_probability_frag = 0
        second_max_probability_frag = 0
        blob_probablities = {}
        frag_probablities = {}

        for code, ijk_list in fragment_xyz.iteritems():
            markers = np.ones(cut_map_orth.shape)
            for i, j, k in ijk_list:
                markers[i, j, k] = 0

            dist = ndimage.distance_transform_edt(markers)
            fragment_mask = dist < max_radii_in_pixels
            common_count = np.nansum(mask & fragment_mask)
            code_prob = 100.0 * common_count / all_counts  # if all_counts != 0 else 0
            fragment_mask_counts = np.nansum(fragment_mask)
            frag_prob = 100.0 * common_count / fragment_mask_counts  # if fragment_mask_counts != 0 else 0
            blob_probablities[code] = code_prob
            frag_probablities[code] = frag_prob

            self.logger.debug("ijk=({},{},{}) shape=({},{},{}), atom_xyz=({},{},{}), code_prob={}, frag_prob={}".format(
                i, j, k, cut_map_orth.shape[0], cut_map_orth.shape[1], cut_map_orth.shape[2],
                fragment_atom[code].position[0], fragment_atom[code].position[1], fragment_atom[code].position[2],
                code_prob, frag_prob
            ))
            self.logger.debug("chain={}, res={}, fragment={}, element={}".format(
                fragment_atom[code].chain_id, fragment_atom[code].res_name, fragment_atom[code].fragment_id,
                fragment_atom[code].element
            ))
            if False and MATPLOTLIB:
                plt.imshow(markers[i, :, :], cmap='hot', interpolation='nearest')
                plt.show()
                plt.imshow(markers[:, j, :], cmap='hot', interpolation='nearest')
                plt.show()
                plt.imshow(markers[:, :, k], cmap='hot', interpolation='nearest')
                plt.show()

            if max_probability < code_prob and code_prob > self.options.MINIMAL_SEGMENT_PROBABILITY:
                second_max_probability = max_probability
                max_probability = code_prob

                second_max_probability_frag = max_probability_frag
                max_probability_frag = frag_prob

                fragment_id = fragment_atom[code].fragment_id
                chain_id = fragment_atom[code].chain_id
                res_name = fragment_atom[code].res_name
                res = fragment_atom[code].get_fragment()

        if self.options.DEBUG_MORE:
            for code, code_prob in sorted(blob_probablities.items(), key=operator.itemgetter(1), reverse=True):
                print 'sp->', code, code_prob

        self.format_probabilities(result_data, blob_probablities, "blob_coverage",
                                  self.options.MINIMAL_SEGMENT_PROBABILITY, frag_probablities, "res_coverage")

        self.logger.debug('get_res_name_with_spheres took %s s' % ((datetime.datetime.now() - start).total_seconds()))
        return chain_id, res_name, fragment_id, max_probability, second_max_probability, max_probability_frag, second_max_probability_frag, res

    def get_threshold(self, density_std_threshold=None):
        # 6 sigma seems to be too high since it should be 6 solvent sigma
        if 'FoFc_std' in self.result_global_data:
            if density_std_threshold is None:
                density_std_threshold = self.options.DENSITY_STD_THRESHOLD
            return density_std_threshold * self.result_global_data['FoFc_std']  # + self.result_global_data['FoFc_mean']
        raise Exception('FoFc_std is not calculated')

    def print_skeleton(self, point_list):
        # options = np.get_printoptions()
        # np.set_printoptions(precision=3, threshold=float("inf"))
        # data_str = str(np.array(point_list))
        # np.set_printoptions(**options)
        data_str = "[%s]" % ",".join(["[%.3f,%.3f,%.3f]" % point for point in point_list])
        return data_str

    def describe_graph(self, blob, result_data):
        start = datetime.datetime.now()

        result_data["skeleton_data"] = self.print_skeleton(blob.skeleton)

        graph = blob.skeleton_graph(self.options.BLOB_GRAPH_DIST)
        graph = blob.prune_deg_3(graph, 1.5 * self.options.BLOB_GRAPH_DIST)
        descriptors = blob.graph_descriptors(graph)
        for key, value in descriptors.iteritems():
            result_data["skeleton_" + key] = value

        self.logger.info('describe_graph took %ss' % (datetime.datetime.now() - start).total_seconds())

    def analyze_blobs(self):
        blob_done_count = 0
        output_blobs = []

        threshold = self.get_threshold()

        start = datetime.datetime.now()
        blobls = self.get_blobs(self.options.MIN_VOLUME_LIMIT, threshold, self.options.BLOB_MERGE_DIST)
        self.logger.info('Get blobs took %s' % (datetime.datetime.now() - start))
        self.logger.info('%d blobs found bigger than %f' % (len(blobls), self.options.MIN_VOLUME_LIMIT))

        for i_blob, blob in enumerate(blobls):
            ############ FLUS LOGGERS
            [h.flush() for h in self.logger.handlers]
            #############
            result_data = DataAggregator()

            # get numpy array in orth
            min_xyz, max_xyz = self.get_blob_min_max_in_orth(blob)
            cut_map_orth, min_xyz_out, max_xyz_out = self.do_cut_in_orth(min_xyz, max_xyz)

            cut_map_orth_part = self.mask_other_parts(
                cut_map_orth,
                min_xyz, max_xyz,
                blob.local_maxi_o,
                threshold,
                title='%s iblob: %s' % (self.pdb_code, i_blob),
                blob=blob,
            )

            closest_chain_id, closest_res_name, closest_fragment_id, max_probability, second_max_probability, max_probability_frag, second_max_probability_frag, res = self.get_res_name_with_spheres(
                cut_map_orth_part, threshold, min_xyz_out, max_xyz_out, result_data, self.options.DEFAULT_SPHERE_RADIUS)

            # zrezygnowalismy bo ekstremalnie wolno dziala
            if self.options.WALKER_MIN_SEGMENTATION_LIMIT > 0.0:
                if max_probability < self.options.WALKER_MIN_SEGMENTATION_LIMIT:
                    closest_chain_id, closest_res_name, closest_fragment_id, max_probability, res = self.get_most_probable_res_name(
                        cut_map_orth_part, threshold, min_xyz_out, max_xyz_out, result_data)
                else:
                    self.format_probabilities(result_data, {}, "res_walker_prob")

            self.logger.info('%s %s %s v=%.2f prob=%.2f' % (
                closest_res_name, closest_chain_id, closest_fragment_id, blob.volume, max_probability))

            if closest_res_name in KEEP_RESIDUES:
                self.logger.info('   MAINCHAIN recognized %s' % result_data.get("res_coverage", ""))
            elif closest_res_name in ('HOH', 'H2O'):
                self.logger.info('    HOH recognized %s' % result_data.get("res_coverage", ""))
            else:
                if closest_res_name is None or max_probability < self.options.MINIMAL_SEGMENT_PROBABILITY:
                    self.logger.info('    UNKNOWN recognized %s' % result_data.get("res_coverage", ""))
                    closest_chain_id = '?'
                    closest_res_name = 'BLOB'
                    closest_fragment_id = blob_done_count
                    res = None
                    if self.rerefine:
                        continue
                else:
                    self.logger.info('    %s recognized %s' % (closest_res_name, result_data.get("res_coverage", "")))

                blob_done_count = blob_done_count + 1
                output_blobs.append(blob)
                title = "%s %s %s %s" % (str(self.pdb_code), closest_res_name, closest_fragment_id, closest_chain_id)
                result_data['title'] = title
                result_data['pdb_code'] = self.pdb_code
                result_data['res_name'] = closest_res_name
                result_data['res_id'] = closest_fragment_id
                result_data['chain_id'] = closest_chain_id
                result_data['blob_volume_coverage'] = max_probability / 100.0
                result_data['blob_volume_coverage_second'] = second_max_probability / 100.0
                result_data['res_volume_coverage'] = max_probability_frag / 100.0
                result_data['res_volume_coverage_second'] = second_max_probability_frag / 100.0
                if self.options.EDSTATS:
                    self.get_edstats('', None, result_data)
                self.get_atoms_count(res, result_data)

                self.describe_graph(blob, result_data=result_data)
                self.calc_stats(cut_map_orth, np.ones(cut_map_orth.shape), np.nan, result_data)

                # save the part of the that was cat
                if self.options.SAVE_CUT_MAP is True:
                    raise Exception("frac coordinates not in blob")
                    tmp_map_fo = self.temporary['map_fo'] % (self.pdb_code, '0', i_blob, 'BLOB')
                    min_xyz_f, max_xyz_f = self.get_blob_min_max_in_frac(blob)
                    cut_map = self.working_map.cut_map(min_xyz_f, max_xyz_f)
                    cut_map.save(tmp_map_fo)

                cut_map_orth = self.mask_other_parts(
                    cut_map_orth,
                    min_xyz, max_xyz,
                    blob.local_maxi_o,
                    threshold,
                    title='%s closest to %s %s iblob: %s (%s)' % (
                        self.pdb_code, closest_res_name, closest_chain_id, i_blob, result_data.get("res_coverage", "")),
                    blob=blob,
                )

                self.logger.info("blob with %s maxes and %s local maxes volume=%s" % (
                    len(blob.max_point_box_o_list), len(blob.local_maxi_o), blob.volume))

                # get atoms for density masking
                cut_map_orth = self.mask_other_residues(
                    cut_map_orth,
                    -1,
                    -1,
                    self.transformed_position_cut,
                    self.transformed_fragment_id_cut,
                    self.transformed_chain_id_cut,
                    self.atoms_list_cut,
                    min_xyz, max_xyz,
                    self.result_global_data['FoFc_std'],
                    self.logger,
                    title='%s closest to %s %s iblob: %s (%s)' % (
                        self.pdb_code, closest_res_name, closest_chain_id, i_blob, result_data.get("res_coverage", "")),
                    result_data=result_data,
                )

                cut_map_orth, mask, parts_count = self.cut_small_parts(cut_map_orth, threshold)

                try:
                    self.process_blob(cut_map_orth, mask, parts_count, title, result_data)
                except Exception as err:
                    self.logger.info("ERROR " + str(err))
                    continue

        if blob_done_count == 0:
            result_data = DataAggregator()
            result_data.save_results(self.filenames['result'])
        self.result_global_data.save_results(self.filenames['global_result'])

        return output_blobs

    def plot_FoFc(self):
        data = self.Fo_minus_Fc_p1.flatten()
        plt.hist(data, 150, normed=1, facecolor='crimson')
        plt.title("%s DELFWT map; mean: %.3f std: %.3f" % (
            str(self.pdb_code),
            self.result_global_data['FoFc_mean'],
            self.result_global_data['FoFc_std'],
        ))
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(self.filenames['FoFc_histogram'], dpi=150)
        plt.clf()

    def plot_density(self):
        data = self.Fo_p1.flatten()
        plt.hist(data, 150, normed=1, facecolor='blue')
        plt.title("%s FP; solvent radius:%.1f A" % (str(self.pdb_code), self.options.SOLVENT_RADIUS))
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(self.filenames['Fo_density_histogram'], dpi=150)
        plt.clf()

        data = self.two_Fo_minus_Fc_p1.flatten()
        plt.hist(data, 150, normed=1, facecolor='blue')
        plt.title("%s FWT; solvent radius:%.1f A" % (str(self.pdb_code), self.options.SOLVENT_RADIUS))
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(self.filenames['Fo_density_histogram'], dpi=150)
        plt.clf()

    def plot_modeled(self):
        data = self.Fo_p1[self.modeled_mask].flatten()
        plt.hist(data, 150, normed=1, facecolor='green')
        plt.title("%s FP modeled; solvent radius:%.1f A" % (str(self.pdb_code), self.options.SOLVENT_RADIUS))
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(self.filenames['Fo_atom_histogram'], dpi=150)
        plt.clf()

        data = self.two_Fo_minus_Fc_p1[self.modeled_mask].flatten()
        plt.hist(data, 150, normed=1, facecolor='green')
        plt.title("%s FWT modeled; solvent radius:%.1f A" % (str(self.pdb_code), self.options.SOLVENT_RADIUS))
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(self.filenames['Fo_atom_histogram'], dpi=150)
        plt.clf()

    def calc_void_fit(self):
        # FWT
        void_mask_count = np.nansum(self.void_mask)
        if void_mask_count > 0:
            data = self.two_Fo_minus_Fc_p1[self.void_mask].flatten()
            data_std = np.std(data)
            data_mean = np.mean(data)

            if self.options.DEBUG_MORE is False:
                n, bins = np.histogram(data, 150, density=True)
            else:
                n, bins, patches = plt.hist(data, 150, normed=1, facecolor='chartreuse', label='void density')
                plt.clf()

            binsm = (bins[1:] + bins[:-1]) / 2
            popt, pcov = fit_binormal(binsm, n,
                                      [data_mean - 0.15, data_std / 10.0, data_mean + 0.15, data_std / 10.0, 0.5])

            self.result_global_data['TwoFoFc_void_fit_binormal_mean1'] = popt[0]
            self.result_global_data['TwoFoFc_void_fit_binormal_std1'] = popt[1]
            self.result_global_data['TwoFoFc_void_fit_binormal_mean2'] = popt[2]
            self.result_global_data['TwoFoFc_void_fit_binormal_std2'] = popt[3]
            self.result_global_data['TwoFoFc_void_fit_binormal_scale'] = popt[4]
        else:
            self.result_global_data['TwoFoFc_void_fit_binormal_mean1'] = np.nan
            self.result_global_data['TwoFoFc_void_fit_binormal_std1'] = np.nan
            self.result_global_data['TwoFoFc_void_fit_binormal_mean2'] = np.nan
            self.result_global_data['TwoFoFc_void_fit_binormal_std2'] = np.nan
            self.result_global_data['TwoFoFc_void_fit_binormal_scale'] = np.nan

        solvent_mask_count = np.nansum(self.solvent_mask)
        if solvent_mask_count > 0:
            # 2FoFc # FWT
            data = self.two_Fo_minus_Fc_p1[self.solvent_mask].flatten()
            data_std = np.std(data)
            data_mean = np.mean(data)

            if self.options.DEBUG_MORE is False:
                n, bins = np.histogram(data, 150, density=True)
            else:
                n, bins, patches = plt.hist(data, 150, normed=1, facecolor='red', label='not modeled density')
                plt.clf()
            binsm = (bins[1:] + bins[:-1]) / 2
            popt_normal, pcov_normal = fit_normal(binsm, n, [data_mean, data_std])

            self.result_global_data['TwoFoFc_solvent_fit_normal_mean'] = popt_normal[0]
            self.result_global_data['TwoFoFc_solvent_fit_normal_std'] = popt_normal[1]
        else:
            self.result_global_data['TwoFoFc_solvent_fit_normal_mean'] = np.nan
            self.result_global_data['TwoFoFc_solvent_fit_normal_std'] = np.nan

    def plot_void(self):
        data = self.Fo_p1[self.void_mask].flatten()
        data_std = np.std(data)
        data_mean = np.mean(data)

        n, bins, patches = plt.hist(data, 150, normed=1, facecolor='chartreuse', label='void density')
        plt.title("%s FP void density histogram; solvent radious:%.1f A; opening %.1f A" % (
            str(self.pdb_code),
            self.options.SOLVENT_RADIUS,
            self.options.SOLVENT_OPENING_RADIUS,
        ))
        binsm = (bins[1:] + bins[:-1]) / 2
        popt, pcov = fit_binormal(binsm, n, [data_mean - 0.15, data_std / 10.0, data_mean + 0.15, data_std / 10.0, 0.5])
        y = binormal(binsm, popt[0], popt[1], popt[2], popt[3], popt[4])
        plt.plot(binsm, y, 'k--', linewidth=2,
                 label='binormal fit, m1=%.3f, s1=%.3f,\n                    m2=%.3f, s2=%.3f,\n                    scale=%.3f' % (
                     popt[0], popt[1], popt[2], popt[3], popt[4]))

        plt.legend()
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(self.filenames['Fo_void_histogram'], dpi=150)
        plt.clf()

        self.result_global_data['Fo_void_fit_binormal_mean1'] = popt[0]
        self.result_global_data['Fo_void_fit_binormal_std1'] = popt[1]
        self.result_global_data['Fo_void_fit_binormal_mean2'] = popt[2]
        self.result_global_data['Fo_void_fit_binormal_std2'] = popt[3]
        self.result_global_data['Fo_void_fit_binormal_scale'] = popt[4]

        # FWT
        data = self.two_Fo_minus_Fc_p1[self.void_mask].flatten()
        data_std = np.std(data)
        data_mean = np.mean(data)

        n, bins, patches = plt.hist(data, 150, normed=1, facecolor='chartreuse', label='void density')
        plt.title("%s FWT void density histogram; solvent radious:%.1f A; opening %.1f A" % (
            str(self.pdb_code),
            self.options.SOLVENT_RADIUS,
            self.options.SOLVENT_OPENING_RADIUS,
        ))
        binsm = (bins[1:] + bins[:-1]) / 2
        popt, pcov = fit_binormal(binsm, n, [data_mean - 0.15, data_std / 10.0, data_mean + 0.15, data_std / 10.0, 0.5])
        y = binormal(binsm, popt[0], popt[1], popt[2], popt[3], popt[4])
        plt.plot(binsm, y, 'k--', linewidth=2,
                 label='binormal fit, m1=%.3f, s1=%.3f,\n                    m2=%.3f, s2=%.3f,\n                    scale=%.3f' % (
                     popt[0], popt[1], popt[2], popt[3], popt[4]))

        plt.legend()
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(self.filenames['2FoFc_void_histogram'], dpi=150)
        plt.clf()

        self.result_global_data['TwoFoFc_void_fit_binormal_mean1'] = popt[0]
        self.result_global_data['TwoFoFc_void_fit_binormal_std1'] = popt[1]
        self.result_global_data['TwoFoFc_void_fit_binormal_mean2'] = popt[2]
        self.result_global_data['TwoFoFc_void_fit_binormal_std2'] = popt[3]
        self.result_global_data['TwoFoFc_void_fit_binormal_scale'] = popt[4]

    def plot_solvent(self):
        data = self.Fo_p1[self.solvent_mask].flatten()
        data_std = np.std(data)
        data_mean = np.mean(data)

        n, bins, patches = plt.hist(data, 150, normed=1, facecolor='red', label='not modeled density')
        plt.title("%s FP bulk density histogram; solvent radious:%.1f A; opening %.1f A" % (
            str(self.pdb_code),
            self.options.SOLVENT_RADIUS,
            self.options.SOLVENT_OPENING_RADIUS,
        ))
        plt.xticks(list(drange(-0.7, 0.701, 0.1)))
        plt.xlim(-0.7, 0.7)
        binsm = (bins[1:] + bins[:-1]) / 2
        # popt, pcov = fit_binormal(binsm, n, [data_mean-0.15, data_std/10.0, data_mean+0.15, data_std/10.0, 0.5])
        # logger.info('PARAMS %s %s %s %s %s %s' % (str(self.pdb_code), data_mean-0.15, data_std, data_mean+0.15, data_std, 0.5))
        # logger.info('PARAMS %s %s %s %s %s %s' % (str(self.pdb_code), popt[0], popt[1], popt[2], popt[3], popt[4]))
        # y = binormal(binsm, popt[0], popt[1], popt[2], popt[3], popt[4])
        # plt.plot(binsm, y, 'k--', linewidth=2, label='binormal fit, m1=%.3f, s1=%.3f,\n                    m2=%.3f, s2=%.3f,\n                    scale=%.3f' % (popt[0], popt[1], popt[2], popt[3], popt[4]))
        #
        # result_global_data['solvent_fit_binormal_mean1'] = popt[0]
        # result_global_data['solvent_fit_binormal_std1'] = popt[1]
        # result_global_data['solvent_fit_binormal_mean2'] = popt[2]
        # result_global_data['solvent_fit_binormal_std2'] = popt[3]
        # result_global_data['solvent_fit_binormal_scale'] = popt[4]

        y = normal(binsm, data_mean, data_std)
        plt.plot(binsm, y, 'g-', linewidth=2, label='normal from data, m=%.3f, s=%.3f' % (data_mean, data_std))
        popt_normal, pcov_normal = fit_normal(binsm, n, [data_mean, data_std])
        y = normal(binsm, popt_normal[0], popt_normal[1])
        plt.plot(binsm, y, 'b-', linewidth=2,
                 label='fit normal from data, m=%.3f, s=%.3f' % (popt_normal[0], popt_normal[1]))

        plt.legend()
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(self.filenames['Fo_solvent_histogram'], dpi=150)
        plt.clf()

        self.result_global_data['Fo_solvent_fit_normal_mean'] = popt_normal[0]
        self.result_global_data['Fo_solvent_fit_normal_std'] = popt_normal[1]

        # 2FoFc # FWT
        data = self.two_Fo_minus_Fc_p1[self.solvent_mask].flatten()
        data_std = np.std(data)
        data_mean = np.mean(data)

        n, bins, patches = plt.hist(data, 150, normed=1, facecolor='red', label='not modeled density')
        plt.title("%s FWT bulk density histogram; solvent radious:%.1f A; opening %.1f A" % (
            str(self.pdb_code),
            self.options.SOLVENT_RADIUS,
            self.options.SOLVENT_OPENING_RADIUS,
        ))
        plt.xticks(list(drange(-0.7, 0.701, 0.1)))
        plt.xlim(-0.7, 0.7)
        binsm = (bins[1:] + bins[:-1]) / 2

        y = normal(binsm, data_mean, data_std)
        plt.plot(binsm, y, 'g-', linewidth=2, label='normal from data, m=%.3f, s=%.3f' % (data_mean, data_std))
        popt_normal, pcov_normal = fit_normal(binsm, n, [data_mean, data_std])
        y = normal(binsm, popt_normal[0], popt_normal[1])
        plt.plot(binsm, y, 'b-', linewidth=2,
                 label='fit normal from data, m=%.3f, s=%.3f' % (popt_normal[0], popt_normal[1]))

        plt.legend()
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(self.filenames['2FoFc_solvent_histogram'], dpi=150)
        plt.clf()

        self.result_global_data['TwoFoFc_solvent_fit_normal_mean'] = popt_normal[0]
        self.result_global_data['TwoFoFc_solvent_fit_normal_std'] = popt_normal[1]

    def plot_all_solvent(self):
        all_data = self.Fo_p1.flatten()
        modeled = self.Fo_p1[self.modeled_mask]
        solvent = self.Fo_p1[self.solvent_mask]
        void = self.Fo_p1[self.void_mask]

        plt.hist([all_data, modeled, solvent, void], 80, range=(-1.0, 2.5),
                 color=['blue', 'green', 'red', 'chartreuse'], label=['all', 'chain', 'solvent', 'void'],
                 histtype='bar', linewidth=0, rwidth=1.0)
        plt.title("%s; solvent radius:%.1f A" % (str(self.pdb_code), self.options.SOLVENT_RADIUS))
        plt.legend()
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(self.filenames['Fo_all_histogram'], dpi=150)
        plt.clf()

        all_data = self.two_Fo_minus_Fc_p1.flatten()
        modeled = self.two_Fo_minus_Fc_p1[self.modeled_mask]
        solvent = self.two_Fo_minus_Fc_p1[self.solvent_mask]
        void = self.two_Fo_minus_Fc_p1[self.void_mask]

        plt.hist([all_data, modeled, solvent, void], 80, range=(-1.0, 2.5),
                 color=['blue', 'green', 'red', 'chartreuse'], label=['all', 'chain', 'solvent', 'void'],
                 histtype='bar', linewidth=0, rwidth=1.0)
        plt.title("%s; solvent radius:%.1f A" % (str(self.pdb_code), self.options.SOLVENT_RADIUS))
        plt.legend()
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(self.filenames['2FoFc_all_histogram'], dpi=150)
        plt.clf()


def calculate(code, pdb_file=None, cif_file=None, mtz_file=None, pdb_out_data_dir=None, overwrite=False,
              logging_level=None, output_stats=False, task=None, fo='FP', sig_fo='SIGFP', rerefine=True):
    start = datetime.datetime.now()

    logger = logging.getLogger('calculate.%s' % code)
    if logging_level is not None:
        logger.setLevel(logging_level)
        if logging_level > logging.DEBUG:
            ConsoleOutput.disable()
    stderr_handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(stderr_handler)
    logger.info('Processing %s on %s', code, platform.node())

    options = Options()
    data_processor = GatherData(code, pdb_file, cif_file, mtz_file, options, logger, pdb_out_data_dir, overwrite,
                                rerefine)

    logger.info(data_processor.filenames['mtz'])

    if task is not None:
        task.update_state(None, 'REFMAC')
    data_processor.calculate_maps(options.MAX_RESOLUTION_LIMIT, fo=fo, sig_fo=sig_fo, do_cut=rerefine)
    ref_time = (datetime.datetime.now() - start).total_seconds()
    if task is not None:
        task.update_state(None, 'PREPROCESSING')
    data_processor.calculate_map_stats()

    logger.info('%s: GLOBAL MAP VALUES, MEAN %.3f, STD %.3f, MIN: %.3f, MAX: %.3f' % (
        code,
        data_processor.result_global_data['FoFc_mean'],
        data_processor.result_global_data['FoFc_std'],
        data_processor.result_global_data['FoFc_min'],
        data_processor.result_global_data['FoFc_max'],
    ))

    if options.DIFF_MAP is True:
        data_processor.plot_FoFc()

    if options.SOLVENT_STATS:
        data_processor.get_solvent_mask(data_processor.two_Fo_minus_Fc, data_processor.atoms_position)
        data_processor.calculate_solvent_stats()

        if options.SOLVENT_PLOT is True:
            data_processor.plot_density()
            data_processor.plot_modeled()
            data_processor.plot_solvent()
            data_processor.plot_all_solvent()
            data_processor.plot_void()
        else:
            data_processor.calc_void_fit()

    data_processor.clean_maps()

    if options.BLOB is True:
        if task is not None:
            task.update_state(None, 'BLOBS')
        blobs = data_processor.analyze_blobs()
    else:
        data_processor.analyze_res()

    data_processor.cleanup()

    overall_time = (datetime.datetime.now() - start).total_seconds()
    proc_time = overall_time - ref_time
    logger.info('Calculate took %ss' % overall_time)
    logger.info('Finished')

    if output_stats:
        return ref_time, proc_time, blobs


# Main script #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="calculate.py")
    parser.add_argument(dest="pdb_code", nargs="?", type=str, help="<pdb_code>")
    parser.add_argument("-f", "--files", nargs=3, dest="files", type=str, required=False,
                        help="pdb_path cif_path mtz_path", metavar=('PDB', 'mmCIF', "MTZ"))
    args = parser.parse_args()

    if args.pdb_code is not None and args.files is None:
        calculate(args.pdb_code)
    elif args.files is not None and args.pdb_code is None:
        pdb = args.files[0]
        cif = args.files[1]
        mtz = args.files[2]
        code = os.path.splitext(os.path.basename(pdb))[0]
        calculate(code, pdb, cif, mtz)
    else:
        parser.print_help()
        sys.exit(2)
