# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>

import os
import sys
import getopt

import util
import logging
import evaluation as ev
import preprocessing as prep

from classifiers import SamplingClassifier
from subsecting_rfe import SubsectingRFE

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.multiclass import OutputCodeClassifier, OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from mlxtend.classifier import StackingCVClassifier
import imblearn.over_sampling as imbos
import lightgbm as lgb

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

SEED = 23
CLASSES = 200
CLF_CPUS = -1
GRID_CPUS = 1
STACKING_CPUS = 1

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data'))
DATASET_PATH = os.path.join(DATA_FOLDER, "all_summary.csv")
VALIDATION_DATASET_PATH = os.path.join(DATA_FOLDER, "validation_all.csv")
NONXRAY_LIST_PATH = os.path.join(DATA_FOLDER, "non_xray_pdbs.csv")
TWIGHLIGHT_PATH = os.path.join(DATA_FOLDER, "twilight-2017-01-11.tsv")
CMB_PATH = os.path.join(DATA_FOLDER, "cmb.pkl")
TERWILLIGER_PATH = os.path.join(DATA_FOLDER, "terwilliger.pkl")
CAROLAN_PATH = os.path.join(DATA_FOLDER, "carolan.pkl")

CLASS_ATTRIBUTE = "res_name"
SELECTION = ["resolution", "percent_cut", "part_00_electrons", "part_00_volume",

             "electrons_over_resolution_00", "volume_over_resolution_00", "std_over_resolution_00",

             "part_00_mean",
             "delta_std_2", "delta_std_1",
             "local_near_cut_count_N", "local_near_cut_count_O", "local_near_cut_count_C",
             "local_mean", "local_electrons",
             "part_00_density_CI", "part_00_shape_CI",

             "delta_electrons_1", "delta_electrons_2",
             "delta_volume_1", "delta_volume_2",
             "delta_mean_1", "delta_mean_2",
             "delta_skewness_1", "delta_skewness_2",

             "part_00_shape_sqrt_E1", "part_00_shape_sqrt_E3", "part_00_shape_E2_E1",
             "part_00_density_sqrt_E2", "part_00_density_sqrt_E1", "part_00_density_sqrt_E3",
             "part_00_density_E2_E1", "part_00_density_E3_E1", "part_00_density_E3_E2",
             "delta_density_sqrt_E1_1", "delta_density_sqrt_E2_1", "delta_density_sqrt_E3_1",

             "shape_segments_count_over_volume_00", "skewness_over_volume_00",

             "part_00_shape_Z_1_0", "part_00_shape_Z_2_0", "part_00_shape_Z_2_1", "part_00_shape_Z_3_1",
             "part_00_shape_Z_4_0", "part_00_shape_Z_4_2",
             "part_00_shape_FL", "part_00_shape_O5_norm",

             "part_00_density_I6", "part_00_density_I2_norm", "part_00_density_I5_norm", "part_00_density_I4_norm",
             "part_00_density_O3_norm", "part_00_density_O4_norm",
             "part_00_density_Z_3_0", "part_00_density_Z_7_0", "part_00_density_Z_5_0", "part_00_density_Z_6_0",

             "std_over_volume_01", "local_cut_by_mainchain_volume",
             "delta_shape_segments_count_2",
             ]

def get_cmb_data():
    training_data = prep.DatasetCleaner(
        prep.read_dataset(DATASET_PATH),
        class_attribute=CLASS_ATTRIBUTE,
        filter_examples=True,
        max_num_of_classes=CLASSES,
        min_examples_per_class=None,
        unique_attributes=["title"],
        keep="largest",
        validation_data=VALIDATION_DATASET_PATH,
        twilight_data=TWIGHLIGHT_PATH,
        combine_ligands=True,
        remove_symmetry_ligands=True,
        remove_poor_quality_data=True,
        remove_poorly_covered=True,
        nonH_atom_range=(2, 1000000000),
        seed=SEED,
        select_attributes=SELECTION + [CLASS_ATTRIBUTE],
        resolution_range=(0.0, 4.0),
        non_xray_pdb_list=NONXRAY_LIST_PATH,
    )

    return training_data


def get_terwilliger_data():
    training_data = prep.DatasetCleaner(
        prep.read_dataset(DATASET_PATH),
        class_attribute=CLASS_ATTRIBUTE,
        filter_examples=True,
        max_num_of_classes=CLASSES,
        min_examples_per_class=None,
        unique_attributes=["title"],
        keep="largest",
        combine_ligands=True,
        remove_symmetry_ligands=False,
        validation_data=VALIDATION_DATASET_PATH,
        remove_poor_quality_data=False,
        remove_poorly_covered=False,
        nonH_atom_range=(6, 150),
        seed=SEED,
        select_attributes=SELECTION + [CLASS_ATTRIBUTE],
        resolution_range=None,
        non_xray_pdb_list=NONXRAY_LIST_PATH,
    )

    return training_data


def get_carolan_data():
    training_data = prep.DatasetCleaner(
        prep.read_dataset(DATASET_PATH),
        class_attribute=CLASS_ATTRIBUTE,
        filter_examples=True,
        max_num_of_classes=CLASSES,
        min_examples_per_class=None,
        unique_attributes=["title"],
        keep="largest",
        combine_ligands=False,
        remove_symmetry_ligands=False,
        validation_data=VALIDATION_DATASET_PATH,
        remove_poor_quality_data=False,
        min_electron_pct=0.0,
        remove_poorly_covered=False,
        seed=SEED,
        select_attributes=SELECTION + [CLASS_ATTRIBUTE],
        ligand_selection=["017", "1PE", "2GP", "2PE", "5GP", "A3P", "ACO", "ADE", "ADN", "ADP", "AKG", "AMP", "ATP",
                          "B3P", "BCL", "BTB", "BTN", "C2E", "CAM", "CDL", "CHD", "CIT", "CLA", "CMP", "COA", "CXS",
                          "CYC", "DIO", "DTT", "EPE", "F3S", "FAD", "FMN", "FPP", "GOL", "GSH", "H4B", "HC4", "HEA",
                          "HED", "HEM", "IMD", "IPH", "LDA", "MES", "MLI", "MLT", "MPD", "MTE", "MYR", "NAD", "NAP",
                          "NCO", "NHE", "OLA", "ORO", "P6G", "PEG", "PEP", "PG4", "PGA", "PGO", "PHQ", "PLM", "PLP",
                          "POP", "PYR", "RET", "SAM", "SF4", "SIA", "SO4", "SPO", "STU", "TAM", "THP", "TLA", "TPP",
                          "TRS", "TYD", "U10", "UPG"],
        resolution_range=(1.0, 2.5),
        non_xray_pdb_list=NONXRAY_LIST_PATH,
    )

    return training_data


def create_datasets():
    logging.info("Preparing CMB dataset...")
    preprocessed_data = get_cmb_data()
    util.save_model(preprocessed_data, CMB_PATH, compress=5)
    preprocessed_data.data_frame.to_csv(CMB_PATH[:-3] + "csv", index=True, header=True)
    logging.info("Finished preparing CMB dataset...")
    logging.info("-----------------------------------------")
    logging.info("")
    logging.info("")

    logging.info("Preparing Terwilliger dataset...")
    preprocessed_data = get_terwilliger_data()
    util.save_model(preprocessed_data, TERWILLIGER_PATH, compress=5)
    preprocessed_data.data_frame.to_csv(TERWILLIGER_PATH[:-3] + "csv", index=True, header=True)
    logging.info("Finished preparing Terwilliger dataset...")
    logging.info("-----------------------------------------")
    logging.info("")
    logging.info("")

    logging.info("Preparing Carolan dataset...")
    preprocessed_data = get_carolan_data()
    util.save_model(preprocessed_data, CAROLAN_PATH, compress=5)
    preprocessed_data.data_frame.to_csv(CAROLAN_PATH[:-3] + "csv", index=True, header=True)
    logging.info("Finished preparing Carolan dataset...")
    logging.info("-----------------------------------------")


# Selected classifiers #
knn = KNeighborsClassifier(n_jobs=CLF_CPUS, n_neighbors=50, weights="distance", p=1)
rf = RandomForestClassifier(random_state=SEED, n_jobs=CLF_CPUS, n_estimators=150, class_weight=None, max_features=0.4)
lgbm = lgb.LGBMClassifier(objective="multiclass", seed=SEED, nthread=CLF_CPUS, num_leaves=128, learning_rate=0.05,
                   n_estimators=281, min_child_weight=13, min_child_samples=1, min_split_gain=0, subsample=1,
                   colsample_bytree=0.85, scale_pos_weight=1, silent=True)

voter = VotingClassifier(estimators=[('knn', knn),('rf', rf),('lgbm', lgbm)], voting="soft")
voter_weighted = VotingClassifier(estimators=[('knn', knn),('rf', rf),('lgbm', lgbm)], voting="soft", weights=[1, 1, 1.5])
linear_stacker = StackingCVClassifier(classifiers=[rf, knn, lgbm], use_probas=True, random_state=SEED, n_folds=5,
                                      meta_classifier=LogisticRegression(random_state=SEED, n_jobs=CLF_CPUS,
                                                                         max_iter=5000, solver="liblinear",
                                                                         multi_class="ovr"))
non_linear_stacker = StackingCVClassifier(classifiers=[rf, knn, lgbm], use_probas=True, random_state=SEED, n_folds=5,
                                          use_features_in_secondary=True,
                                          meta_classifier=lgb.LGBMClassifier(objective="multiclass", seed=SEED,
                                                                             nthread=CLF_CPUS, num_leaves=128,
                                                                             learning_rate=0.05, n_estimators=281,
                                                                             min_child_weight=13, min_child_samples=1,
                                                                             min_split_gain=0, subsample=1,
                                                                             colsample_bytree=0.85, scale_pos_weight=1,
                                                                             silent=True))


selected_classifiers = [
    Pipeline([("preprocessor", prep.BlobPreprocessor(validation_data=VALIDATION_DATASET_PATH)), ("scaler", MinMaxScaler()), ("clf", knn)]),
    Pipeline([("preprocessor", prep.BlobPreprocessor(validation_data=VALIDATION_DATASET_PATH)), ("scaler", MinMaxScaler()), ("clf", rf)]),
    Pipeline([("preprocessor", prep.BlobPreprocessor(validation_data=VALIDATION_DATASET_PATH)), ("scaler", MinMaxScaler()), ("clf", lgbm)]),
]

stacking = [
    Pipeline([("preprocessor", prep.BlobPreprocessor(validation_data=VALIDATION_DATASET_PATH)), ("scaler", MinMaxScaler()), ("clf", non_linear_stacker)]),
]

stacking_carolan = [
    Pipeline([("preprocessor", prep.BlobPreprocessor(validation_data=VALIDATION_DATASET_PATH, remove_outliers=False, remove_poor_quality=False)), ("scaler", MinMaxScaler()), ("clf", non_linear_stacker)]),
]

# Classifier settings #
grid = {
    # GaussianNB():
    # [{}],
    #
    # DecisionTreeClassifier(random_state=SEED):
    # [{"max_features": [None, 10, 15, 16, 17, 18, 19, 20],
    #   "max_depth": [None, 5, 8, 10, 12, 15],
    #   "criterion": ["gini", "entropy"],
    #   "class_weight": [None, "balanced"]}],
    #
    # KNeighborsClassifier(n_jobs=CLF_CPUS):
    # [{"n_neighbors": [10, 20, 30, 40, 50], # 50
    #   "weights": ["distance"],
    #   "p": [1]}],
    #
    # RandomForestClassifier(random_state=SEED, n_jobs=CLF_CPUS):
    # [{"n_estimators": [150],
    #  "max_features": ["auto", 0.3, 0.4, 0.5], # 0.4
    #  "class_weight": ["balanced", None]}], # None
    #
    # LogisticRegression(random_state=SEED, class_weight="balanced", max_iter=5000, n_jobs=CLF_CPUS,
    #                    solver="liblinear", multi_class="ovr"):
    # [{"C": [10000],
    #   "intercept_scaling": [2]}],
    #
    # SVC(probability=True, random_state=SEED, cache_size=10000, max_iter=5000):
    # [{"C": [1, 10, 100],
    #   "gamma": [1, 10, 100],
    #   "class_weight": ["balanced"]}],
    #
    # SGDClassifier(random_state=SEED, n_iter=100, n_jobs=CLF_CPUS):
    # [{"alpha": [0.000005],
    #   "class_weight": ["balanced", None]}],
    #
    # lgb.LGBMClassifier(objective="multiclass", seed=SEED, nthread=CLF_CPUS):
    # [{
    #     "num_leaves": [64, 128, 256], #128
    #     "min_child_weight": [1, 3, 5, 7, 9, 11, 12, 13, 14, 15], # 13
    #     "min_child_samples": [1],
    #     "min_split_gain": [0],
    #     "subsample": [0.8, 0.9, 1.0], #1
    #     "colsample_bytree": [0.8, 0.85, 0.9, 1.0], #0.85
    #     "scale_pos_weight": [1],
    #     "learning_rate": [0.1], # 0.05
    #     "silent": [True],
    #     "n_estimators": [131] # 281
    # }],
    # StackingCVClassifier(classifiers=[rf, knn, lgbm], random_state=SEED, n_folds=5, use_probas=False,
    #                      meta_classifier=RandomForestClassifier(random_state=SEED, n_jobs=CLF_CPUS, n_estimators=100)):
    # [{
    #     "meta-randomforestclassifier__max_features": ["auto", 0.3],
    #     "meta-randomforestclassifier__class_weight": ["balanced", None],
    # }],
}

preprocessor = {
    prep.BlobPreprocessor(remove_outliers=True,
                          score_outliers=False,
                          remove_poor_quality=True,
                          validation_data=VALIDATION_DATASET_PATH,
                          n_jobs=CLF_CPUS):
    [{}]
}

scaler = {
    MinMaxScaler():
    [{}]
}

# Main script #
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "cmesg")
    except getopt.GetoptError:
        print('run_experiment.py [-c -m -e -s -g]')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-c", "--create_datasets"):
            create_datasets()
            sys.exit()
        elif opt in ("-e", "--evaluate"):
            for dataset_path in [CMB_PATH, TERWILLIGER_PATH, CAROLAN_PATH]:
                training_data = util.load_model(dataset_path)
                X, y = training_data.prepare_for_classification(CLASS_ATTRIBUTE, [CLASS_ATTRIBUTE])

                ev.cross_validate(selected_classifiers, X, y, os.path.basename(dataset_path), training_data,
                                  seed=SEED, cv_folds=10)
        elif opt in ("-g", "--stacked_generalization"):
            for dataset_path in [CMB_PATH, TERWILLIGER_PATH]:
                training_data = util.load_model(dataset_path)
                X, y = training_data.prepare_for_classification(CLASS_ATTRIBUTE, [CLASS_ATTRIBUTE])

                ev.cross_validate(stacking, X, y, os.path.basename(dataset_path), training_data,
                                  seed=SEED, cv_folds=10)
            for dataset_path in [CAROLAN_PATH]:
                training_data = util.load_model(dataset_path)
                X, y = training_data.prepare_for_classification(CLASS_ATTRIBUTE, [CLASS_ATTRIBUTE])

                ev.cross_validate(stacking_carolan, X, y, os.path.basename(dataset_path), training_data,
                                  seed=SEED, cv_folds=10)
            sys.exit()
        elif opt in ("-m", "--model_selection"):
            for dataset_path in [CMB_PATH]:
                training_data = util.load_model(dataset_path)
                X, y = training_data.prepare_for_classification(CLASS_ATTRIBUTE, [CLASS_ATTRIBUTE])

                ev.run_experiment(grid, X, y, os.path.basename(dataset_path), training_data,
                                  pipeline=[("preprocessor", preprocessor), ("scaler", scaler)],
                                  evaluation_metric=metrics.make_scorer(ev.top_n_accuracy, needs_proba=True, top_n=5),
                                  seed=SEED, jobs=GRID_CPUS, repeats=2, folds=5, outer_cv=10)
            sys.exit()
        elif opt in ("-s", "--early_stopping"):
            for dataset_path in [CMB_PATH]:
                training_data = util.load_model(dataset_path)
                X, y = training_data.prepare_for_classification(CLASS_ATTRIBUTE, [CLASS_ATTRIBUTE])
                early_lgbm = lgb.LGBMClassifier(objective="multiclass", seed=SEED, nthread=CLF_CPUS, num_leaves=128,
                                                learning_rate=0.05, n_estimators=3000, min_child_weight=13,
                                                min_child_samples=1, min_split_gain=0, subsample=1,
                                                colsample_bytree=0.85, scale_pos_weight=1, silent=True)
                ev.lgbm_cv_early_stopping(early_lgbm, X, y, pipeline=[("preprocessor", preprocessor), ("scaler", scaler)],
                                      seed=SEED)
        sys.exit()