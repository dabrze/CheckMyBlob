# coding: utf-8

from sklearn.ensemble import IsolationForest

import os
import gc
import time
import json
import warnings
import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing
from util import DatasetStatistics
from sklearn.base import BaseEstimator, TransformerMixin

__author__ = "Marcin Kowiel, Dariusz Brzezinski"


class BlobPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, selected_attributes=None, seed=23, score_outliers=False, remove_outliers=True,
                 remove_poor_quality=True, outlier_fraction=0.005, isolation_tree_sample=2000,
                 isolation_forest_size=200, validation_data=None, n_jobs=-1):
        self.selected_attributes = selected_attributes
        self.seed = seed
        self.score_outliers = score_outliers
        self.remove_outliers = remove_outliers
        self.remove_poor_quality = remove_poor_quality
        self.outlier_fraction = outlier_fraction
        self.isolation_tree_sample = isolation_tree_sample
        self.isolation_forest_size = isolation_forest_size
        self.n_jobs = n_jobs

        self.validation_data = validation_data

        self.isolation_forest = None
        self.column_names = None

    def __repr__(self):
        result = []
        for attr, value in iter(self.__dict__.items()):
            result.append("=".join([attr, str(value)]))

        return '%s(%s)' % (self.__class__.__name__, ", ".join(sorted(result)))

    def fit(self, X, y=None):
        self.isolation_forest = IsolationForest(n_estimators=self.isolation_forest_size,
                                                max_samples=self.isolation_tree_sample,
                                                contamination=self.outlier_fraction,
                                                random_state=self.seed,
                                                n_jobs=self.n_jobs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.remove_outliers:
                self._remove_outliers(X, y)
            elif self.score_outliers:
                self._score_outliers(X, y)

            if self.remove_poor_quality:
                self._filter_low_quality_examples(X, y)

        return self

    def transform(self, X, y=None):
        Xt = X

        if self.score_outliers:
            anomaly_scores = self.isolation_forest.decision_function(X)

        if self.selected_attributes is not None:
            Xt = X.loc[:, self.selected_attributes]

        if self.score_outliers:
            Xt.loc[:, "anomaly_score"] = anomaly_scores

        self.column_names = Xt.columns.values

        return Xt

    def _filter_low_quality_examples(self, X, y):
        if self.validation_data is not None:
            validation_df = read_validation_data(self.validation_data)
            # repeated title, but different values -> multiple conformations
            is_duplicate_title = validation_df.duplicated(subset="title", keep=False)
            validation_df.set_index("title", inplace=True)
            nonidentical_duplicates = validation_df[is_duplicate_title.values].index
            multiple_conformations = nonidentical_duplicates[nonidentical_duplicates.isin(X.index)]
            multiple_conformations_num = multiple_conformations.shape[0]

            if multiple_conformations_num > 0:
                logging.info("Removing %s examples with multiple conformations from training data",
                             str(multiple_conformations_num))
                X.drop(multiple_conformations, inplace=True)
                y.drop(multiple_conformations, inplace=True)

    def _score_outliers(self, X, y):
        self.isolation_forest.fit(X)

    def _remove_outliers(self, X, y):
        gc.collect()
        self._score_outliers(X, y)
        outlier_pred = self.isolation_forest.predict(X)
        outliers = X[outlier_pred == -1].index
        outlier_num = outliers.shape[0]
        gc.collect()

        if outlier_num > 0:
            logging.info("Removing %s outliers from training data", str(outlier_num))
            X.drop(outliers, inplace=True)
            y.drop(outliers, inplace=True)


class DatasetCleaner:
    MAX_R_FACTOR = 0.3
    MIN_OCCUPANCY = 0.3
    POLY_THRESHOLD = 30
    MIN_RSCC = 0.6

    UNKNOWN_LIGANDS = ["UNK", "UNX", "UNL", "DUM"]
    ANY_NUCLEOTYDE = ["N"]
    UNLABELED = ["BLOB", "", "?"]
    PEPTIDES_DNA_RNA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                        'MSE', 'PHE', 'PRO', 'SEC', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
                        'DA', 'DG', 'DT', 'DC', 'DU',
                        'A', 'G', 'T', 'C', 'U', }
    WATER = ["HOH", "H20", "WAT", "DOD"]
    IGNORED_RES_NAMES = set().union(UNKNOWN_LIGANDS, ANY_NUCLEOTYDE, UNLABELED, PEPTIDES_DNA_RNA, WATER)

    KEY_ATTRIBUTE = "title"
    PDBID_ATTRIBUTE = "pdb_code"
    ELECTRON_ATTRIBUTE = "part_00_electrons"
    ILLEGAL_ATTRIBUTES = ["pdb_code", "res_id", "chain_id",
                          "local_res_atom_count", "local_res_atom_non_h_count",
                          "local_res_atom_non_h_occupancy_sum",
                          "local_res_atom_non_h_electron_sum",
                          "local_res_atom_non_h_electron_occupancy_sum",
                          "local_res_atom_C_count",
                          "local_res_atom_N_count", "local_res_atom_O_count",
                          "local_res_atom_S_count",
                          "dict_atom_non_h_count",
                          "dict_atom_non_h_electron_sum", "dict_atom_C_count",
                          "dict_atom_N_count", "dict_atom_O_count",
                          "dict_atom_S_count",
                          "fo_col", "fc_col", "weight_col", "grid_space",
                          "solvent_radius",
                          "solvent_opening_radius",
                          "part_step_FoFc_std_min", "part_step_FoFc_std_max",
                          "part_step_FoFc_std_step",
                          "local_volume", "res_coverage", "blob_coverage",
                          "blob_volume_coverage", "blob_volume_coverage_second",
                          "res_volume_coverage", "res_volume_coverage_second",
                          "skeleton_data"
                          ]

    GLOBALS = ["resolution_max_limit", "part_step_FoFc_std_min",
               "part_step_FoFc_std_max", "part_step_FoFc_std_step"]

    GRAPH_ATTRIBUTES = ["local_maxi_graph_low_cycles", "local_maxi_graph_low_cycle_6", "local_maxi_graph_low_cycle_7",
                        "local_maxi_graph_low_cycle_5", "local_maxi_graph_low_closeness_000_002",
                        "local_maxi_graph_low_closeness_002_004", "local_maxi_graph_low_closeness_004_006",
                        "local_maxi_graph_low_closeness_006_008", "local_maxi_graph_low_closeness_008_010",
                        "local_maxi_graph_low_closeness_010_012", "local_maxi_graph_low_closeness_012_014",
                        "local_maxi_graph_low_closeness_014_016", "local_maxi_graph_low_closeness_016_018",
                        "local_maxi_graph_low_diameter", "local_maxi_graph_low_radius",
                        "local_maxi_graph_low_deg_5_plus", "local_maxi_graph_low_density",
                        "local_maxi_graph_low_periphery", "local_maxi_graph_low_graph_clique_number",
                        "local_maxi_graph_low_nodes", "local_maxi_graph_low_avg_degree", "local_maxi_graph_low_edges",
                        "local_maxi_graph_low_average_clustering", "local_maxi_graph_low_center",
                        "local_maxi_graph_low_deg_4", "local_maxi_graph_low_deg_0",
                        "local_maxi_graph_low_deg_1", "local_maxi_graph_low_deg_2", "local_maxi_graph_low_deg_3",
                        "local_maxi_graph_hi_cycles", "local_maxi_graph_hi_cycle_6", "local_maxi_graph_hi_cycle_7",
                        "local_maxi_graph_hi_cycle_5", "local_maxi_graph_hi_closeness_000_002",
                        "local_maxi_graph_hi_closeness_002_004", "local_maxi_graph_hi_closeness_004_006",
                        "local_maxi_graph_hi_closeness_006_008", "local_maxi_graph_hi_closeness_008_010",
                        "local_maxi_graph_hi_closeness_010_012", "local_maxi_graph_hi_closeness_012_014",
                        "local_maxi_graph_hi_closeness_014_016", "local_maxi_graph_hi_closeness_016_018",
                        "local_maxi_graph_hi_diameter", "local_maxi_graph_hi_radius",
                        "local_maxi_graph_hi_deg_5_plus", "local_maxi_graph_hi_density",
                        "local_maxi_graph_hi_periphery", "local_maxi_graph_hi_graph_clique_number",
                        "local_maxi_graph_hi_nodes", "local_maxi_graph_hi_avg_degree", "local_maxi_graph_hi_edges",
                        "local_maxi_graph_hi_average_clustering", "local_maxi_graph_hi_center",
                        "local_maxi_graph_hi_deg_4", "local_maxi_graph_hi_deg_0",
                        "local_maxi_graph_hi_deg_1", "local_maxi_graph_hi_deg_2", "local_maxi_graph_hi_deg_3",
                        ]

    def __init__(self, data_frame, class_attribute="res_name", filter_examples=False, unique_attributes=None,
                 max_num_of_classes=200, min_examples_per_class=None, drop_attributes=ILLEGAL_ATTRIBUTES + GLOBALS,
                 select_attributes=None, where_title=None, sort_by_title=None, seed=23, drop_parts=range(3, 10),
                 validation_data=None, remove_poor_quality_data=True, keep=None, remove_poorly_covered=True,
                 blob_coverage_threshold=0.1, res_coverage_threshold=0.2, twilight_data=None, combine_ligands=True,
                 remove_symmetry_ligands=True, ligand_selection=None, discretize_add_noise=False,
                 discretize_round_noise=False, min_electron_pct=0.5, nonH_atom_range=None, resolution_range=(0, 4),
                 non_xray_pdb_list=None, edstats_data=None, min_ZOa=None, max_ZDa=None, training_data=True):
        """
        Initializes a new preprocessor object with the specified settings.
        """
        self.data_frame = data_frame
        self.class_series = None
        self.filter_examples = filter_examples
        self.unique_attributes = unique_attributes
        self.max_num_of_classes = max_num_of_classes
        self.min_examples_per_class = min_examples_per_class
        self.drop_attributes = drop_attributes
        self.select_attributes = select_attributes
        self.where_title = where_title
        self.sort_by_title = sort_by_title
        self.label_encoder = preprocessing.LabelEncoder()
        self.class_attribute = class_attribute
        self.seed = seed
        self.drop_parts = drop_parts
        self.validation_data = validation_data
        self.remove_poor_quality_data = remove_poor_quality_data
        self.keep = keep
        self.remove_poorly_covered = remove_poorly_covered
        self.blob_coverage_threshold = blob_coverage_threshold
        self.res_coverage_threshold = res_coverage_threshold
        self.twilight_data = twilight_data
        self.combine_ligands = combine_ligands
        self.remove_symmetry_ligands = remove_symmetry_ligands
        self.ligand_selection = ligand_selection
        self.discretize_add_noise = discretize_add_noise
        self.discretize_round_noise = discretize_round_noise
        self.min_electron_pct = min_electron_pct
        self.nonH_atom_range = nonH_atom_range
        self.resolution_range = resolution_range
        self.non_xray_pdb_list = non_xray_pdb_list
        self.edstats_data = edstats_data
        self.min_ZOa = min_ZOa
        self.max_ZDa = max_ZDa
        self.training_data = training_data

        self.clean()

    @property
    def classes(self):
        """
        Classes of the dataset.
        """
        if len(self.label_encoder.classes_) > 0:
            return self.label_encoder.classes_
        else:
            raise Exception("Data not prepared yet!")

    def __repr__(self):
        result = []
        for attr, value in iter(self.__dict__.items()):
            if attr == "where_title" and self.where_title is not None:
                result.append("=".join([attr, str(True)]))
            elif attr == "drop_attributes":
                if self.drop_attributes == self.ILLEGAL_ATTRIBUTES + self.GLOBALS:
                    result.append("=".join([attr, "ILLEGAL_ATTRIBUTES+GLOBALS"]))
                else:
                    result.append("=".join([attr, str(value)]))
            elif attr != "data_frame" and attr != "class_series":
                result.append("=".join([attr, str(value)]))

        return '%s(%s)' % (self.__class__.__name__, ", ".join(sorted(result)))

    def clean(self):
        logging.info("Cleaning data...")

        if self.training_data:
            logging.info("Initial dataset:\r\n%s", DatasetStatistics(self.data_frame,
                                                                     self.data_frame.loc[:, self.class_attribute]))

            try:
                non_xray_df = read_non_xray_pdb_list(self.non_xray_pdb_list)
                non_xray = self.data_frame[(self.data_frame[self.PDBID_ATTRIBUTE].isin(non_xray_df.loc[:, "pdbid"]))]
                non_xray_num = non_xray.shape[0]
                non_xray_unique = len(pd.unique(non_xray.loc[:, self.PDBID_ATTRIBUTE]))

                if non_xray_num > 0:
                    logging.info(
                        "Removing %s examples taken from PDB entries with experimental methods other tha X-ray "
                        "diffraction (%s non-xray PDB files)", str(non_xray_num), str(non_xray_unique))
                    self.data_frame = self.data_frame.drop(non_xray.index)
            except:
                logging.warning("Could not find list of non-xray pdb files")
                # pdb_code pdbid

        no_electrons = self.data_frame[~(self.data_frame[self.ELECTRON_ATTRIBUTE] > 0)].index
        no_electrons_num = no_electrons.shape[0]
        if no_electrons_num > 0:
            logging.info("Removing %s examples with no electron density", str(no_electrons_num))
            self.data_frame = self.data_frame.drop(no_electrons)

        if self.training_data:
            if self.where_title is not None:
                self.data_frame = self.data_frame.loc[self.data_frame[self.KEY_ATTRIBUTE].isin(self.where_title), :]

            if self.sort_by_title:
                self.data_frame = self.data_frame.sort_values(by=self.KEY_ATTRIBUTE)

            if self.filter_examples:
                self._drop_duplicates(self.unique_attributes, self.keep)
                if self.remove_poorly_covered:
                    self._filter_poorly_covered_examples()
                self._filter_examples(self.max_num_of_classes, self.min_examples_per_class, self.ligand_selection,
                                      self.nonH_atom_range, self.resolution_range)

        self.data_frame.set_index(self.KEY_ATTRIBUTE, inplace=True)
        self._drop_attributes(self.drop_attributes)
        self._drop_parts(self.drop_parts)
        self._feature_engineering()
        self._discretize(add_noise=self.discretize_add_noise, round_noise=self.discretize_round_noise)
        self._zero_nas()
        self._convert_columns_to_floats()
        self._select_attributes(self.select_attributes)

        if self.training_data:
            logging.info("Dataset after preprocessing:\r\n%s",
                         DatasetStatistics(self.data_frame, self.data_frame.loc[:, self.class_attribute]))

    def prepare_for_classification(self, selected_class_attribute, all_class_attributes=None):
        """
        Prepares the dataset for training and/or testing
        :param selected_class_attribute: the attribute that will be used during learning
        :type selected_class_attribute: str
        :param all_class_attributes: all possible class attributes in the dataset. If not given defaults
        to the selected class attribute. This list ensures that attributes that contain direct information about the
        true class will not be used during training.
        :type all_class_attributes: list of str
        :return: the dataset divided into a data frame with unlabeled examples and a data frame (series) only with
        labels
        """
        gc.collect()
        logging.info("Preparing data for classification...")

        if all_class_attributes is None:
            all_class_attributes = [selected_class_attribute]

        self.class_series = self.data_frame[selected_class_attribute]
        drop_class_attributes = [attr for attr in all_class_attributes if attr in self.data_frame.columns]
        self.data_frame = self.data_frame.drop(drop_class_attributes, axis=1)

        self.label_encoder.fit(self.class_series)
        labels_frame = pd.Series(self.label_encoder.transform(self.class_series), self.data_frame.index)
        gc.collect()

        return self.data_frame, labels_frame

    def _feature_engineering(self):
        delta_attributes = ["electrons", "std", "skewness", "mean", "volume", "parts", "shape_segments_count",
                            "density_segments_count", "density_sqrt_E1", "density_sqrt_E2", "density_sqrt_E3"]
        for i in range(1, 3):
            for delta_attribute in delta_attributes:
                try:
                    self.data_frame.loc[:, "delta_" + delta_attribute + "_" + str(i)] = \
                        self.data_frame.loc[:, "part_0" + str(i - 1) + "_" + delta_attribute] - \
                        self.data_frame.loc[:, "part_0" + str(i) + "_" + delta_attribute]
                    gc.collect()
                except Exception as ex:
                    logging.warning("Feature engineering: %s", ex)

        over_attributes = ["electrons", "std", "skewness", "shape_segments_count", "density_segments_count"]
        for i in range(0, 3):
            for over_attribute in over_attributes:
                try:
                    self.data_frame.loc[:, over_attribute + "_over_volume_0" + str(i)] = \
                        self.data_frame.loc[:, "part_0" + str(i) + "_" + over_attribute] / \
                        self.data_frame.loc[:, "part_0" + str(i) + "_volume"]
                    gc.collect()
                except Exception as ex:
                    logging.warning("Feature engineering: %s", ex)

        over_attributes = ["volume", "electrons", "std", "skewness"]
        for i in range(0, 3):
            for over_attribute in over_attributes:
                try:
                    self.data_frame.loc[:, over_attribute + "_over_resolution_0" + str(i)] = \
                        self.data_frame.loc[:, "part_0" + str(i) + "_" + over_attribute] / \
                        self.data_frame.loc[:, "resolution"]
                    gc.collect()
                except Exception as ex:
                    logging.warning("Feature engineering: %s", ex)

        self.data_frame.loc[:, "percent_cut"] = self.data_frame.loc[:, "local_cut_by_mainchain_volume"] / \
                                                (self.data_frame.loc[:, "local_cut_by_mainchain_volume"] +
                                                 self.data_frame.loc[:, "part_00_volume"])
        gc.collect()

    def _discretize(self, add_noise=False, round_noise=False):
        self.data_frame.loc[:, "resolution"] = self.data_frame.loc[:, "resolution"].round(decimals=1)
        self.data_frame.loc[:, "local_std"] = self.data_frame.loc[:, "local_std"].round(decimals=2)

        if add_noise:
            noise = np.random.uniform(-0.15, 0.15, self.data_frame.shape[0])
            if round_noise:
                noise = noise.round(decimals=1)
            self.data_frame.loc[:, "resolution"] = self.data_frame.loc[:, "resolution"] + noise

            noise = np.random.uniform(-0.015, 0.015, self.data_frame.shape[0])
            if round_noise:
                noise = noise.round(decimals=2)
            self.data_frame.loc[:, "local_std"] = self.data_frame.loc[:, "local_std"] + noise

    def _drop_duplicates(self, subset, keep="largest"):
        # Leave one row from a set of identical rows
        self.data_frame = self.data_frame.drop_duplicates(keep="first")

        if subset is None or not subset or keep is None:
            return

        if keep == "largest":
            if subset != [self.KEY_ATTRIBUTE]:
                warnings.warn("Leaving largest volume when filtering by something different than the key attribute")
            self.data_frame = self.data_frame.sort_values(by=[self.KEY_ATTRIBUTE, "part_00_volume"], ascending=False)
            self.data_frame = self.data_frame.drop_duplicates(subset=subset, keep="first")
        else:
            self.data_frame = self.data_frame.drop_duplicates(subset=subset, keep=keep)

    def _filter_poorly_covered_examples(self):
        poorly_covered = self.data_frame[self.data_frame.blob_volume_coverage < self.blob_coverage_threshold].index
        poorly_covered_num = poorly_covered.shape[0]

        if poorly_covered_num > 0:
            logging.info("Removing %s examples with blobs covered by the model below %s%%", str(poorly_covered_num),
                         str(self.blob_coverage_threshold * 100))
            self.data_frame = self.data_frame.drop(poorly_covered)

        res_poorly_covered = self.data_frame[self.data_frame.res_volume_coverage < self.res_coverage_threshold].index
        res_poorly_covered_num = res_poorly_covered.shape[0]

        if res_poorly_covered_num > 0:
            logging.info("Removing %s examples with models covered by the blob below %s%%", str(res_poorly_covered_num),
                         str(self.res_coverage_threshold * 100))
            self.data_frame = self.data_frame.drop(res_poorly_covered)

    def _res_threshold_attributes(self, attributes, res_threshold, dummy_value=-10):
        res_thresholded = self.data_frame.resolution > res_threshold
        res_thresholded_num = np.sum(res_thresholded)

        if res_thresholded_num > 0:
            logging.info("Thresholding graph attributes for %d examples" % res_thresholded_num)
            self.data_frame.loc[res_thresholded, attributes] = dummy_value

    def _calculate_top_n_classes(self, max_num_of_classes):
        """
        Calculates the the top n most frequent classes.
        :param max_num_of_classes: maximum number of expected classes
        :type max_num_of_classes: int
        :return: list of max_num_of_classes classes
        """
        if max_num_of_classes <= 0:
            raise Exception("The number of classes cannot be smaller than 1!")

        gc.collect()
        classes = self.data_frame.loc[:, self.class_attribute].copy()
        gc.collect()
        res_name_count = classes.value_counts()
        gc.collect()
        res_name_count.sort_values(inplace=True, ascending=False)
        if res_name_count.__len__() < max_num_of_classes:
            raise Exception("Not enough classes in the dataset!")

        return res_name_count.iloc[:max_num_of_classes].index.values

    def _filter_examples(self, max_num_of_classes, min_examples_per_class, ligand_selection, nonH_atom_range,
                         resolution_range):
        """
        Excludes unnecessary and underrepresented class examples from the dataset.
        :param max_num_of_classes: maximum number of expected classes after filtering (used when
        min_examples_per_class=None)
        :type max_num_of_classes: int
        :param min_examples_per_class: minimum number of examples per class (underepresented classes will be filtered
        out)
        :type min_examples_per_class: int
        """
        if nonH_atom_range is not None:
            min_atoms = nonH_atom_range[0]
            max_atoms = nonH_atom_range[1]
            poor_atom_count = (self.data_frame.dict_atom_non_h_count < min_atoms) | \
                              (self.data_frame.dict_atom_non_h_count > max_atoms)
            poor_atom_count_num = np.sum(poor_atom_count)

            if poor_atom_count_num > 0:
                logging.info("Removing %s examples with non-H atom count < %s or > %s",
                             str(poor_atom_count_num), str(min_atoms), str(max_atoms))
                self.data_frame = self.data_frame.drop(self.data_frame[poor_atom_count].index)

        if resolution_range is not None:
            min_res = resolution_range[0]
            max_res = resolution_range[1]
            poor_res = (self.data_frame.resolution < min_res) | (self.data_frame.resolution > max_res)
            poor_res_num = np.sum(poor_res)

            if poor_res_num > 0:
                logging.info("Removing %s examples with resolution < %s or > %s",
                             str(poor_res_num), str(min_res), str(max_res))
                self.data_frame = self.data_frame.drop(self.data_frame[poor_res].index)

        # Potential res_name N/A problem
        if np.sum(self.data_frame.loc[:, self.class_attribute].isnull()) > 0:
            raise Exception("N/A class values in dataset!")

        if ligand_selection is not None:
            logging.info("Selecting only user-defined ligands")
            self.data_frame = self.data_frame[self.data_frame.loc[:, self.class_attribute].isin(ligand_selection)]
            return

        # non-ligands and unknown ligands
        ignored_ligands = self.data_frame.loc[:, self.class_attribute].isin(self.IGNORED_RES_NAMES)
        logging.info("Removing %d unknown and non-ligand structures", np.sum(ignored_ligands))
        self.data_frame = self.data_frame[~ignored_ligands]

        if self.combine_ligands:
            logging.info("Creating ligand complexes")
            res_names = self.data_frame.loc[:, self.class_attribute].unique()
            self.data_frame.loc[:, self.class_attribute] = self.data_frame.apply(self._detect_polyligand, axis=1,
                                                                                 args=(res_names, self.POLY_THRESHOLD,
                                                                                       self.IGNORED_RES_NAMES))
            mislabeled_ligands = self.data_frame[self.class_attribute] == "_DELETE_"
            mislabeled_ligands_num = np.sum(mislabeled_ligands)

            if mislabeled_ligands_num > 0:
                logging.info("Removing %d poorly covered ligand complexes", mislabeled_ligands_num)
                self.data_frame = self.data_frame.drop(self.data_frame[mislabeled_ligands].index)

        if self.remove_symmetry_ligands:
            self.data_frame.loc[:, "is_symmetry_ligand"] = self.data_frame.res_coverage.str[1:-1].apply(
                lambda x: self._detect_symmetry_ligands(x)
            )
            symmetry_ligand_num = np.sum(self.data_frame.is_symmetry_ligand)

            if symmetry_ligand_num > 0:
                logging.info("Removing %d ligands centered in a symmetry", symmetry_ligand_num)
                self.data_frame = self.data_frame.drop(self.data_frame[self.data_frame.is_symmetry_ligand].index)

            self.data_frame = self.data_frame.drop("is_symmetry_ligand", axis=1)

        if self.remove_poor_quality_data:
            # removing examples with extremely poor quality
            # ligands modeles without over half of the non-H electrons
            poor_electrons = self.data_frame.local_res_atom_non_h_electron_sum \
                             < self.min_electron_pct * self.data_frame.dict_atom_non_h_electron_sum
            poor_electrons_num = np.sum(poor_electrons)

            if poor_electrons_num > 0:
                logging.info("Removing %s ligands without over %d %% of non-H electrons modeled",
                             str(poor_electrons_num), int(100 * self.min_electron_pct))
                self.data_frame = self.data_frame.drop(self.data_frame[poor_electrons].index)

            if self.validation_data is not None:
                # repeated title, but different values -> multiple conformations
                validation_df = read_validation_data(self.validation_data)
                validation_df = validation_df.drop_duplicates(subset="title", keep=False)

                joined_data = self.data_frame.merge(validation_df, on="title", how="left")
                poor_r_factor = joined_data.EDS_R > self.MAX_R_FACTOR
                poor_occupancy = joined_data.avgoccu < self.MIN_OCCUPANCY
                poor_rscc = joined_data.rscc < self.MIN_RSCC
                poor_quality = joined_data[poor_r_factor | poor_occupancy | poor_rscc]

                poor_r_factor_num = np.sum(poor_r_factor)
                poor_occupancy_num = np.sum(poor_occupancy)
                poor_rscc_num = np.sum(poor_rscc)
                poor_quality_num = poor_quality.shape[0]

                if poor_quality_num > 0:
                    logging.info("Removing %d examples with R > %s (%d) or occupancy < %s (%d) or RSCC < %s (%d)",
                                 poor_quality_num, self.MAX_R_FACTOR, poor_r_factor_num,
                                 self.MIN_OCCUPANCY, poor_occupancy_num, self.MIN_RSCC, poor_rscc_num)
                    self.data_frame = self.data_frame[~self.data_frame.title.isin(poor_quality.title)]

            if self.twilight_data is not None:
                # repeated title, but different values -> multiple conformations
                twilight_df = read_twilight_data(self.twilight_data)
                twilight_df = twilight_df.drop_duplicates(subset="title", keep=False)

                joined_data = self.data_frame.merge(twilight_df, on="title", how="left")
                poor_quality = joined_data[joined_data.Valid == "Y"]
                poor_quality_num = poor_quality.shape[0]

                if poor_quality_num > 0:
                    logging.info("Removing %d examples flagged by Twilight (%s)", poor_quality_num,
                                 os.path.basename(self.twilight_data))
                    self.data_frame = self.data_frame[~self.data_frame.title.isin(poor_quality.title)]

            if self.edstats_data is not None:
                edstats_df = read_edstats_data(self.edstats_data)
                edstats_df = edstats_df.drop_duplicates(subset="title", keep=False)

                joined_data = self.data_frame.merge(edstats_df, on="title", how="left")
                poor_ZOa = joined_data.ZOa < self.min_ZOa
                poor_ZDa = joined_data.ZDa >= self.max_ZDa
                poor_quality = joined_data[poor_ZOa | poor_ZDa]

                poor_ZOa_num = np.sum(poor_ZOa)
                poor_ZDa_num = np.sum(poor_ZDa)
                poor_quality_num = poor_quality.shape[0]

                if poor_quality_num > 0:
                    logging.info("Removing %d examples with ZOa < %s (%d) "
                                 "or ZDa >= %s (%d)",
                                 poor_quality_num, self.min_ZOa, poor_ZOa_num,
                                 self.max_ZDa, poor_ZDa_num)
                    self.data_frame = self.data_frame[~self.data_frame.title.isin(poor_quality.title)]

        # minimal number of examples/maximum number of classes
        if min_examples_per_class is not None:
            logging.info("Limiting dataset to classes with at least %d examples", min_examples_per_class)
            count_attribute = self.class_attribute + "_count"
            res_name_count = self.data_frame.loc[:, self.class_attribute].value_counts()
            self.data_frame.loc[:, count_attribute] = self.data_frame.loc[:, self.class_attribute] \
                .map(res_name_count).astype(int)
            self.data_frame = self.data_frame[self.data_frame[count_attribute] >= min_examples_per_class]
            self.data_frame = self.data_frame.drop(count_attribute, axis=1)
        else:
            logging.info("Limiting dataset to %d most popular classes", max_num_of_classes)
            gc.collect()
            top_n_classes = self._calculate_top_n_classes(max_num_of_classes)
            gc.collect()
            classes = self.data_frame.loc[:, self.class_attribute].copy()
            isin_vector = classes.isin(top_n_classes)
            gc.collect()
            self.data_frame = self.data_frame[isin_vector]
            gc.collect()

    def _drop_attributes(self, attributes):
        """
        Drops selected attributes and makes the data frame smaller.
        :param attributes: list of attributes to be dropped
        :type attributes: list of str
        """
        if attributes is not None:
            self.data_frame = self.data_frame.drop(attributes, axis=1)

    def _drop_parts(self, parts):
        """
        Drops selected parts and makes the data frame smaller.
        :param parts: list of part numbers
        :type parts: list of int
        """
        if parts is not None:
            attributes = []
            for part in parts:
                attributes.extend([col for col in list(self.data_frame) if col.startswith("part_0" + str(part))])
            self.data_frame = self.data_frame.drop(attributes, axis=1)

    def _select_attributes(self, attributes):
        """
        Leaves selected attributes and makes the data frame smaller.
        :param attributes: list of attributes to be left after filtering
        :type attributes: list of str
        """
        if attributes is not None:
            self.data_frame = self.data_frame.loc[:, attributes]

    def _zero_nas(self):
        """
        Turns NA values into zeros.
        """
        self.data_frame = self.data_frame.fillna(0.0)

    def _convert_columns_to_floats(self):
        """
        Converts all columns into floats and limits their values to the float range.
        """
        tmp_frame = self.data_frame.loc[:, [self.class_attribute]]
        for col in list(tmp_frame):
            try:
                self.data_frame = self.data_frame.drop(col, axis=1)
            except ValueError as ex:
                logging.warning(ex)

        self.data_frame[self.data_frame > 10e32] = 10e32
        self.data_frame[self.data_frame < -10e32] = -10e32

        self.data_frame = self.data_frame.astype("float32")
        self.data_frame = pd.concat([self.data_frame, tmp_frame], axis=1)

    @staticmethod
    def _combine_symmetries(res_coverage_dict):
        new_dict = dict()
        for key, value in res_coverage_dict.items():
            modified_key = ":".join(key.split(":")[:-1])
            new_dict[modified_key] = max(new_dict.setdefault(modified_key, value), value)

        return new_dict

    @staticmethod
    def _detect_symmetry_ligands(res_coverage_col):
        new_res_dict = dict()
        res_coverage_dict = json.loads(res_coverage_col)

        for key, value in res_coverage_dict.items():
            modified_key = ":".join(key.split(":")[:-1])
            new_res_dict[modified_key] = max(new_res_dict.setdefault(modified_key, value), value)

        return len(new_res_dict) == 1 and len(res_coverage_dict) > 1

    @staticmethod
    def _detect_polyligand(row, res_names, coverage_threshold, ignored_res):
        from operator import itemgetter

        res_coverage_col = row["res_coverage"][1:-1]

        res_coverage_dict = DatasetCleaner._combine_symmetries(json.loads(res_coverage_col))
        sorted_keys = sorted(((k.split(":")[0], v) for k, v in res_coverage_dict.items()
                              if (k.split(":")[0] not in ignored_res)))

        if len(sorted_keys) == 1:
            return sorted_keys[0][0]
        else:
            new_res_name = "_".join((k for k, v in sorted_keys if (v >= coverage_threshold)))

            if new_res_name == "":
                new_res_name = max(sorted_keys, key=itemgetter(1))[0]

            if "_" not in new_res_name and new_res_name not in res_names:
                new_res_name = "_DELETE_"

            return new_res_name


def read_dataset(path):
    logging.info("Reading: %s", os.path.basename(path))
    start = time.time()
    df_header = pd.read_csv(path, sep=";", header=0, na_values=["n/a", "nan", ""],
                            keep_default_na=False, engine="c", nrows=1)
    string_cols = ["title", "pdb_code", "res_id", "res_name", "chain_id", "fo_col", "fc_col", "res_coverage",
                   "blob_coverage", "skeleton_data"]
    float_cols = {c: np.float64 for c in list(df_header) if c not in string_cols}
    for string_col in string_cols:
        float_cols[string_col] = "str"

    df = pd.read_csv(path, sep=";", header=0, na_values=["n/a", "nan", ""],
                     keep_default_na=False, engine="c", dtype=float_cols)
    logging.info("Read dataset in: %.2f seconds", time.time() - start)
    return df


def read_merge_datasets(paths):
    dfs = []

    for path in paths:
        dfs.append(read_dataset(path))

    return pd.concat(dfs, ignore_index=True)


def read_validation_data(path):
    validation_data = pd.read_csv(path, sep=",", header=0, na_values=["n/a", "nan", ""],
                                  keep_default_na=False, engine="c")
    validation_data = validation_data.drop_duplicates(keep="first")
    float_cols = ["rsr_nonHOH_normed", "rscc_nonHOH_normed", "EDS_R", "avgoccu",
                  "ligRSRZ", "mogul_angles_rmsz", "mogul_bonds_rmsz", "rscc"]
    for float_col in float_cols:
        validation_data[float_col] = validation_data[float_col].astype(np.float64)

    return validation_data


def read_twilight_data(path):
    twilight_data = pd.read_csv(path, sep="\t", header=0, na_values=["n/a"],
                                keep_default_na=False, engine="c", low_memory=False)
    twilight_data = twilight_data.drop_duplicates(keep="first")

    twilight_data.loc[:, "chain_id"] = twilight_data.ResNr.str[0].str.strip()
    twilight_data.loc[:, "res_id"] = twilight_data.ResNr.str[1:].str.strip()
    twilight_data.loc[:, "title"] = twilight_data.PDBID.str.cat([twilight_data.LigNm, twilight_data.res_id,
                                                                 twilight_data.chain_id], sep=" ")

    return twilight_data


def read_edstats_data(path):
    twilight_data = pd.read_csv(path, sep=",", header=0, na_values=["n/a"],
                                keep_default_na=False, engine="c",
                                low_memory=False)

    return twilight_data


def read_non_xray_pdb_list(path):
    non_xray_pdbs = pd.read_csv(path, header=0, na_values=["n/a"], keep_default_na=False, engine="c",
                                low_memory=False)

    return non_xray_pdbs
