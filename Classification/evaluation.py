# coding: utf-8

import os
import gc
import copy
import csv
import time
import logging
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import sklearn.model_selection as skms
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.base import clone
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.fixes import bincount
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.classification import _prf_divide
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

import preprocessing as prep
import util

__author__ = 'Marcin Kowiel, Dariusz Brzezinski'


class Metrics:
    def __init__(self, y_true, y_pred, y_prob, y_resolution, resolution_range=None):
        if resolution_range is not None:
            subset = np.arange(0, y_true.shape[0])[(y_resolution > resolution_range[0]) & (y_resolution <= resolution_range[1])]
            y_true = y_true.iloc[subset]
            y_pred = y_pred[subset]
            if y_prob is not None:
                y_prob = y_prob[subset]

        self.resolution_range = resolution_range
        self.subset_size = y_true.shape[0]
        self.n_classes = y_true.nunique()
        self.accuracy = metrics.accuracy_score(y_true, y_pred) if y_true is not None else -1
        self.top_5_accuracy = top_n_accuracy(y_true, y_prob, top_n=5) if y_true is not None else -1
        self.top_10_accuracy = top_n_accuracy(y_true, y_prob, top_n=10) if y_true is not None else -1
        self.top_20_accuracy = top_n_accuracy(y_true, y_prob, top_n=20) if y_true is not None else -1
        self.macro_recall = metrics.recall_score(y_true, y_pred, average="macro") if y_true is not None else -1
        self.kappa = metrics.cohen_kappa_score(y_true, y_pred) if y_true is not None else -1
        self.gmean = g_mean(y_true, y_pred) if y_true is not None else -1
        self.brier = brier_score_loss_for_true_class(y_true, y_prob) if y_true is not None else -1
        self.worst_prediction_rank = worst_prediction_rank(y_true, y_prob) if y_true is not None else -1


class Evaluation:
    """
    Evaluation results for a given classifier on a given dataset.
    """
    def __init__(self, dataset_name, preprocessing, classifier, search_params, y_true, y_pred, y_prob, y_resolution,
                 search_results, evaluation_metric, training_time, testing_time,
                 resolution_ranges=[(0.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0)]):
        """
        Constructor.
        :param dataset_name: Dataset name
        :type dataset_name: string
        :param preprocessing: Global preprocessing applied to the dataset.
        :type preprocessing: Preprocessing
        :param classifier: Evaluated classifier
        :type classifier: BaseEstimator
        :param search_params: Parameter grid used if model selection was performed
        :type: search_params: dict
        :param y_true: True class labels for test data
        :type y_true: list
        :param y_pred: Predicted class labels for test data
        :type y_pred: list
        :param y_prob: Predicted probabilities of each class for each example
        :type y_prob: array-like
        :param search_results: Grid search results
        :type search_results: GridSearchCV
        :param evaluation_metric: Evaluation metric used to select best model during grid search
        :type evaluation_metric: string
        :param processing_time: Evaluation time in seconds
        :type processing_time: int
        """
        self.dataset_name = dataset_name
        self.dataset_stats = util.DatasetStatistics(preprocessing.data_frame, preprocessing.class_series)
        self.preprocessing = preprocessing
        self.classifier = classifier
        self.search_params = search_params
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.y_resolution = y_resolution
        self.resolution_ranges = resolution_ranges
        self.search_results = search_results
        self.best_std = search_results.cv_results_["std_test_score"][search_results.best_index_] \
            if search_results is not None else None
        self.evaluation_metric = evaluation_metric
        self.training_time = training_time
        self.testing_time = testing_time

        self.metrics = Metrics(y_true, y_pred, y_prob, y_resolution, resolution_range=None)
        self.resolution_metrics = []

        if resolution_ranges is not None and len(resolution_ranges) > 0:
            for resolution_range in resolution_ranges:
                self.resolution_metrics.append(Metrics(y_true, y_pred, y_prob, y_resolution, resolution_range))

        self.num_of_classes = self.dataset_stats.num_of_classes
        self.start_date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()-training_time-testing_time))
        self.end_date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def __repr__(self):
        """
        Returns a string representation of an evaluation. Basic statistics in a readable form.
        :return: evaluation score names and values
        """

        return "Results:\r\n" \
               + "\tStart: {0}".format(self.start_date_time) + "\r\n" \
               + "\tEnd: {0}".format(self.end_date_time) + "\r\n" \
               + "\tTraining time: {0:.0f} s".format(self.training_time) + "\r\n" \
               + "\tTesting time: {0:.0f} s".format(self.testing_time) + "\r\n" \
               + "\tEvaluation metric: {0}".format(self.evaluation_metric) + "\r\n" \
               + "\tCV params: {0}".format(self.search_results.best_params_
                                           if self.search_results is not None else None) + "\r\n" \
               + "\tAverage CV score: {0:.3f}".format(self.search_results.best_score_
                                                      if self.search_results is not None else -1) + "\r\n" \
               + "\tCV std: {0:.3f}".format(self.best_std if self.best_std is not None else -1) + "\r\n" \
               + "\tAccuracy: {0:.3f}".format(self.metrics.accuracy) + "\r\n" \
               + "\tTop-5 accuracy: {0:.3f}".format(self.metrics.top_5_accuracy) + "\r\n" \
               + "\tTop-10 accuracy: {0:.3f}".format(self.metrics.top_10_accuracy) + "\r\n" \
               + "\tTop-20 accuracy: {0:.3f}".format(self.metrics.top_20_accuracy) + "\r\n" \
               + "\tMacro recall: {0:.3f}".format(self.metrics.macro_recall) + "\r\n" \
               + "\tKappa: {0:.3f}".format(self.metrics.kappa) + "\r\n" \
               + "\tG-mean: {0:.3f}".format(self.metrics.gmean) + "\r\n" \
               + "\tBrier score: {0:.3f}".format(self.metrics.brier) + "\r\n" \
               + "\tWorst prediction rank: {0}/{1}".format(self.metrics.worst_prediction_rank, str(self.num_of_classes))

    def write_to_csv(self, file_name="ExperimentResults.csv",
                     save_to_folder=os.path.join(os.path.dirname(__file__), "ExperimentResults"), fold_num=None):
        """
        Adds a new row to a csv file with evaluation results. If the given filenmae does not correspond to any existing
        csv, a new file is created.
        :param file_name: csv file name
        :type file_name: string
        :param save_to_folder: folder to save the file to
        :type save_to_folder: string, optional (default=source file folder/ExperimentResults)
        """
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
                header = ["Start date",
                          "End date",
                          "Dataset",
                          "Fold",
                          "Examples",
                          "Attributes",
                          "Number of classes",
                          "Min class examples",
                          "Max class examples",
                          "Classes",
                          "Preprocessing",
                          "Grid",
                          "Grid metric",
                          "Pipeline",
                          "Classifier",
                          "Training time",
                          "Testing time",
                          "Best CV parameters",
                          "Best average CV score",
                          "CV scores",
                          "CV standard deviation"]

                metric_names = ["Accuracy", "Top 5 acc.", "Top 10 acc.", "Top 20 acc.", "Macro recall", "Kappa",
                                "G-mean", "Brier score", "Worst prediction rank"]
                header.extend(metric_names)
                for resolution_metric in self.resolution_metrics:
                    header.extend([m + " " + str(resolution_metric.resolution_range) for m in metric_names])

                writer.writerow(header)

            row = [self.start_date_time,
                   self.end_date_time,
                   self.dataset_name,
                   fold_num if fold_num is not None else "",
                   self.dataset_stats.examples,
                   self.dataset_stats.attributes,
                   self.dataset_stats.num_of_classes,
                   self.dataset_stats.min_examples,
                   self.dataset_stats.max_examples,
                   " ".join([str(key) + ": " + str(value)
                             for key, value in self.dataset_stats.classes.iteritems()]),
                   self.preprocessing,
                   self.search_params,
                   self.evaluation_metric,
                   util.deep_repr(self.classifier).replace('\n', ' ').replace('\r', ''),
                   util.classifier_repr(self.classifier).replace('\n', ' ').replace('\r', ''),
                   self.training_time,
                   self.testing_time,
                   self.search_results.best_params_ if self.search_results is not None else "",
                   self.search_results.best_score_ if self.search_results is not None else "",
                   self.search_results.cv_results_ if self.search_results is not None else "",
                   self.best_std]

            for metrics in [self.metrics] + self.resolution_metrics:
                row.extend([metrics.accuracy, metrics.top_5_accuracy, metrics.top_10_accuracy, metrics.top_20_accuracy,
                            metrics.macro_recall, metrics.kappa, metrics.gmean, metrics.brier,
                            metrics.worst_prediction_rank])

            writer.writerow(row)

    def save_predictions(self, save_to_folder=os.path.join(os.path.dirname(__file__), "ExperimentResults")):
        try:
            y_true_prob = self.y_prob[np.arange(len(self.y_prob)), self.y_true]\
                if self.y_prob is not None else pd.DataFrame(np.zeros((len(self.y_true), 1)))
        except:
            y_true_prob = 0
        y_pred_prob = self.y_prob[np.arange(len(self.y_prob)), self.y_pred] \
            if self.y_prob is not None else pd.DataFrame(np.zeros((len(self.y_true), 1)))

        classes = self.y_prob.shape[1]
        sorted_pred = np.argsort(self.y_prob, axis=1)
        rank = classes - np.apply_along_axis(np.where, 1, np.equal(sorted_pred, np.repeat(self.y_true[:, np.newaxis],
                                                                                          classes, 1))).flatten()

        rscc = np.full((len(self.y_true), 1), np.nan).flatten()

        if self.preprocessing.validation_data is not None:
            try:
                validation_df = prep.read_validation_data(self.preprocessing.validation_data)
            except:
                DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data'))
                VALIDATION_DATASET_PATH = os.path.join(DATA_FOLDER, "validation_all.csv")
                validation_df = prep.read_validation_data(VALIDATION_DATASET_PATH)

            validation_df = validation_df.drop_duplicates(subset="title", keep="first")

            rscc = pd.DataFrame({"title": self.y_true.index, "pred": self.y_true.values}).merge(validation_df, on="title", how="left").loc[:, "rscc"].values

        predictions = pd.DataFrame({"y_true": self.preprocessing.classes[self.y_true],
                                    "y_pred": self.preprocessing.classes[self.y_pred],
                                    "y_true_prob": y_true_prob,
                                    "y_pred_prob": y_pred_prob,
                                    "prob_diff": y_pred_prob - y_true_prob,
                                    "is_correct": self.y_true == self.y_pred,
                                    "rank": rank,
                                    "resolution": self.y_resolution,
                                    "rscc": rscc},
                                   index=self.y_true.index,
                                   columns=["y_pred", "y_true", "y_pred_prob", "y_true_prob",
                                            "prob_diff", "is_correct", "rank", "resolution", "rscc"])
        predictions.to_csv(os.path.join(save_to_folder, _get_file_name(self.dataset_name, self.classifier,
                                                                       "predictions", "csv")))

    def save_model(self, save_to_folder=os.path.join(os.path.dirname(__file__), "ExperimentResults")):
        util.save_model(self.classifier, os.path.join(save_to_folder, _get_file_name(self.dataset_name, self.classifier, "model", "pkl")))

    def save_feature_importance(self, plot=False, save_to_folder=os.path.join(os.path.dirname(__file__),
                                                                        "ExperimentResults")):
        sns.set_style("whitegrid")

        classifier = self.classifier.steps[-1][1]
        classifier_name = str(classifier)[:str(classifier).index("(")]
        column_names = list(self.preprocessing.data_frame)

        try:
            blobber = self.classifier.named_steps['preprocessor']
            column_names = blobber.column_names
        except KeyError:
            logging.warning("No BlobberPreprocessor in pipeline when attempting to plot feature importance.")

        try:
            rfecv = self.classifier.named_steps['rfe']
            column_names = column_names[rfecv.support_]
        except KeyError:
            pass

        try:
            importances = classifier.feature_importances_
        except AttributeError:
            try:
                importances = np.absolute(classifier.coef_).sum(axis=0)/np.absolute(classifier.coef_).sum()
            except (ValueError, AttributeError):
                logging.warning("Classifier does not support feature importance")
                return

        indices = np.argsort(importances)[::-1]

        importance_df = pd.DataFrame(
            {"attribute": column_names[indices],
             "importance": importances[indices]},
            index=range(len(column_names)),
            columns=["attribute", "importance"])
        importance_df.to_csv(os.path.join(save_to_folder, _get_file_name(self.dataset_name, self.classifier, "feature_importance", "csv")),
                             index=False)
        if plot:
            self._plot_interactive_feature_importance(column_names[indices], importances[indices], classifier_name)

    def save_confusion_matrix(self, save_to_folder=os.path.join(os.path.dirname(__file__), "ExperimentResults")):
        """
        Saves a confusion matrix (numpy array) to a file
        :param save_to_folder: folder to save the file to
        :type save_to_folder: string, optional (default=source file folder/ExperimentResults)
        :return:
        """
        np.savetxt(os.path.join(save_to_folder, _get_file_name(self.dataset_name, self.classifier, "confusion_matrix", "txt")),
                   metrics.confusion_matrix(self.y_true, self.y_pred).astype("int"), fmt="%d")
        with open(os.path.join(save_to_folder, _get_file_name(self.dataset_name, self.classifier, "confusion_matrix", "txt")) +
                  "_classes.txt", 'w') as file_obj:
            for c in self.preprocessing.classes:
                file_obj.write(c + '\n')

    def plot_confusion_matrix(self, save_to_folder=os.path.join(os.path.dirname(__file__), "ExperimentResults")):
        """
        Plots a confusion matrix based on the evalution results.
        :param save_to_folder: determines the folder where the plot should be saved
        :return:  plt, file_name
        """
        confusion_matrix = metrics.confusion_matrix(self.y_true, self.y_pred)
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Greys)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(self.preprocessing.classes))
        plt.xticks(tick_marks, self.preprocessing.classes, rotation=90)
        plt.yticks(tick_marks, self.preprocessing.classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        file_name = _get_file_name(self.dataset_name, self.classifier, "confusion_matrix", "png")
        if not os.path.exists(save_to_folder):
            os.mkdir(save_to_folder)
        plt.savefig(os.path.join(save_to_folder, file_name))
        plt.close()

    def plot_interactive_confusion_matrix(self, save_to_folder=os.path.join(os.path.dirname(__file__),
                                                                            "ExperimentResults")):
        """
        Plots an interactive confusion matrix based on the evalution results.
        :param save_to_folder: determines the folder where the plot should be saved
        :return:  plt, file_name
        """
        import plotly.offline
        import plotly.graph_objs as go

        confusion_matrix = metrics.confusion_matrix(self.y_true, self.y_pred)
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        file_name = os.path.join(save_to_folder, _get_file_name(self.dataset_name, self.classifier, "confusion_matrix", "html"))
        title = "<b>Acc: {:.1f}%, T5: {:.1f}%, T10: {:.1f}%, T20: {:.1f}%, " \
                "MR: {:.1f}%, K: {:.1f}%, " \
                "G: {:.1f}%</b>"\
            .format(self.metrics.accuracy*100, self.metrics.top_5_accuracy*100,
                    self.metrics.top_10_accuracy*100, self.metrics.top_20_accuracy*100,
                    self.metrics.macro_recall*100, self.metrics.kappa*100, self.metrics.gmean*100)

        for resolution_metric in self.resolution_metrics:
            title += "<br>{:s}:{:05d}:{:03d} Acc: {:04.1f}%, T5: {:04.1f}%, T10: {:04.1f}%, T20: {:04.1f}%, " \
                     "MR: {:04.1f}%, K: {:04.1f}%, G: {:04.1f}%"\
                .format(resolution_metric.resolution_range, resolution_metric.subset_size,  resolution_metric.n_classes,
                        resolution_metric.accuracy*100, resolution_metric.top_5_accuracy*100,
                        resolution_metric.top_10_accuracy*100, resolution_metric.top_20_accuracy*100,
                        resolution_metric.macro_recall*100, resolution_metric.kappa*100, resolution_metric.gmean*100)

        data = [
            go.Heatmap(
                x=self.preprocessing.classes,
                y=self.preprocessing.classes[::-1],
                z=cm_normalized[::-1, :].round(3),
                colorscale=[[0.0, "rgb(255, 255, 255)"], [1.0, "rgb(0, 0,0)"]]
            )
        ]

        layout = go.Layout(
            titlefont={"size": 14},
            title=title,
            xaxis={"title": "Predicted label"},
            yaxis={"title": "True label"},
            width=1000,
            height=775,
            autosize=False,
            margin=go.Margin(
                t=155,
                l=200,
                r=200,
                autoexpand=False
            ),
        )

        plotly.offline.plot(dict(data=data, layout=layout), filename=file_name)

    def _plot_interactive_feature_importance(self, feature_names, feature_importances, classifier_name,
                                             save_to_folder=os.path.join(os.path.dirname(__file__),
                                                                         "ExperimentResults")):
        """
        Plots an interactive confusion matrix based on the evalution results.
        :param save_plot_to_file: determines whether the created plot should be save to a file
        :param save_to_folder: determines the folder where the plot should be saved
        :return:  plt, file_name
        """
        import plotly.offline
        import plotly.graph_objs as go

        file_name = os.path.join(save_to_folder, _get_file_name(self.dataset_name, self.classifier, "feature_importance", "html"))

        data = [
            go.Bar(
                x=feature_names,
                y=feature_importances
            )
        ]

        layout = go.Layout(
            title="Feature importance for " + classifier_name,
            xaxis={"title": ""},
            yaxis={"title": "Importance"},
            width=800,
            height=600,
            autosize=False
        )

        plotly.offline.plot(dict(data=data, layout=layout), filename=file_name)


class RepeatedStratifiedKFold(skms._split._BaseKFold):
    """
    Repeated Stratified K-Folds cross validation iterator. Provides train/test indices to split data in train test
    sets. This cross-validation object is a variation of KFold that returns stratified folds and repeats the process a
    given number of times. The folds are made by preserving the percentage of samples for each class.
    """

    def _iter_test_indices(self, X=None, y=None, groups=None):
        raise NotImplementedError

    def __init__(self, n_iter=5, n_splits=2, random_state=None):
        """
        :param n_iter: number of iterations (reshuffles)
        :type n_iter: int, default=5
        :param n_splits: number of folds. Must be at least 2.
        :type n_splits: int, default=2
        :param random_state: Pseudo-random number generator state used for random sampling. If None, use default numpy
        RNG for shuffling
        :type random_state: None, int or RandomState
        """
        super(RepeatedStratifiedKFold, self).__init__(n_splits*n_iter, True, random_state)
        self.n_iter = n_iter
        self.skfs = []

        for i in range(n_iter):
            self.skfs.append(skms.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state+i))

    def _iter_test_masks(self, X, y=None, groups=None):
        for i, skf in enumerate(self.skfs):
            test_folds = skf._make_test_folds(X, y)
            for j in range(skf.n_splits):
                yield test_folds == j

    def __repr__(self):
        return '%s.%s(n_iter=%i, n_splits=%i, shuffle=%s, random_state=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n_iter,
            self.n_splits/self.n_iter,
            self.shuffle,
            self.random_state,
        )

    def __len__(self):
        return self.n_splits


def g_mean(y_true, y_pred, labels=None, correction=0.001):
    """
    Computes the geometric mean of class-wise recalls.
    :param y_true: True class labels.
    :type y_true: list
    :param y_pred: Predicted class labels.
    :type y_pred: array-like
    :param labels:  Labels present in the data can be excluded, for example to calculate a multiclass average ignoring a
    majority negative class, while labels not present in the data will result in 0 components in a macro average.
    :type labels: list, optiona
    :param correction: substitution/correction for zero values in class-wise recalls
    :type correction: float
    :return: G-mean value
    """
    present_labels = unique_labels(y_true, y_pred)

    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels, assume_unique=True)])

    le = LabelEncoder()
    le.fit(labels)
    y_true = le.transform(y_true)
    y_pred = le.transform(y_pred)
    sorted_labels = le.classes_

    # labels are now from 0 to len(labels) - 1 -> use bincount
    tp = y_true == y_pred
    tp_bins = y_true[tp]

    if len(tp_bins):
        tp_sum = bincount(tp_bins, weights=None, minlength=len(labels))
    else:
        # Pathological case
        true_sum = tp_sum = np.zeros(len(labels))

    if len(y_true):
        true_sum = bincount(y_true, weights=None, minlength=len(labels))

    # Retain only selected labels
    indices = np.searchsorted(sorted_labels, labels[:n_labels])
    tp_sum = tp_sum[indices]
    true_sum = true_sum[indices]

    recall = _prf_divide(tp_sum, true_sum, "recall", "true", None, "recall")
    recall[recall == 0] = correction

    return sp.stats.mstats.gmean(recall)


def brier_score_loss_for_true_class(y_true, y_proba):
    """
    Calculates Brier score for from a set of class probabilities.
    :param y_true: True class labels.
    :type y_true: list
    :param y_proba: Predicted class probabilities.
    :type y_proba: array-like
    :return: Brier score
    """
    if y_proba is None:
        return -1
    try:
        return metrics.brier_score_loss(y_true, y_proba[np.arange(len(y_proba)), y_true])
    except:
        return -1


def top_n_accuracy(y_true, y_proba, top_n=10):
    """

    :param y_true:
    :param y_proba:
    :param top_n:
    :return:
    """
    if y_proba is None:
        return -1

    try:
        top_n_pred = np.argsort(y_proba, axis=1)[:, -top_n:]
        return np.average(
            np.apply_along_axis(np.any, 1, np.equal(top_n_pred, np.repeat(y_true[:, np.newaxis], top_n, 1))))
    except:
        return -1


def worst_prediction_rank(y_true, y_proba):
    """

    :param y_true:
    :param y_proba:
    :return:
    """
    if y_proba is None:
        return -1

    try:
        classes = y_proba.shape[1]
        sorted_pred = np.argsort(y_proba, axis=1)
        worst_rank = classes - np.min(
            np.apply_along_axis(np.where, 1, np.equal(sorted_pred, np.repeat(y_true[:, np.newaxis], classes, 1))))
        return worst_rank
    except:
        return -1


def mmap(var, filename):
    #temp_folder = tempfile.mkdtemp()
    temp_folder = os.path.abspath(os.path.dirname(__file__))
    dir_filename = os.path.join(temp_folder, '%s.mmap' %filename)
    if os.path.exists(dir_filename):
        os.unlink(dir_filename)
    _ = joblib.dump(var, dir_filename)
    return joblib.load(dir_filename, mmap_mode='r+'), dir_filename


def train_classifier(classifier, X_train, y_train):
    """
    Fits a classifier on the given data.
    :param classifier: classifier
    :param X_train: training attribute values
    :param y_train: training classes
    :return: a fitted model
    """
    gc.collect()
    logging.info("Training...")
    # Experimental - to reduce memory consumption
    # y_train, y_train_filename = mmap(y_train, 'y_train')
    classifier.fit(X_train, y_train)
    # Experimental - to reduce memory consumption
    # X_train, X_train_filename = mmap(X_train, 'X_train')
    gc.collect()

    return classifier


def test_classifier(classifier, X_test, y_test, n_jobs=None):
    """
    Tests a classifier on the given. If possible, this method also gets the probability of each class prediction.
    :param classifier: classifier
    :param X_test: testing attribute values
    :param y_test: testing classes
    :param n_jobs: overwriting number of threads used during prediction
    :return: y_true, y_pred, y_prob
    """
    gc.collect()
    logging.info("Testing...")

    if n_jobs is not None:
        for param in classifier.get_params().keys():
            if "n_jobs" in param or "nthread" in param:
                classifier.set_params(**{param: n_jobs})
                logging.debug("Update %s to %s" % (param, n_jobs))
    # Experimental - to reduce memory consumption
    # X_test, X_test_filename = mmap(X_test, 'X_test')

    y_true = y_test
    y_pred = classifier.predict(X_test)
    y_prob = None
    gc.collect()

    try:
        y_prob = classifier.predict_proba(X_test)
    except:
        logging.debug("Classifier does not produce probability scores")
    gc.collect()

    y_resolution = None
    try:
        y_resolution = X_test.resolution
    except:
        logging.debug("Did not find resolution column in test data")

    return y_true, y_pred, y_prob, y_resolution


def select_model(classifier, param_grid, X_train, y_train, pipeline=None, seed=23, evaluation_metric="accuracy",
                 repeats=5, folds=2, jobs=-1):
    """
    Performs model selection.
    :param classifier: classifier
    :param param_grid: parameters values to choose from
    :param X_train: training attribute values
    :param y_train: training class values
    :param pipeline: preprocessing pipeline steps
    :param seed: random seed for repeated stratified cross-validation
    :param evaluation_metric: metric used to select best model
    :param repeats: number of repeatitions in repeated stratified cross-validation
    :param folds: number of folds in each repetition in repeated stratified cross-validation
    :param jobs: number of jobs to run in parallel.
    :return:
    """
    cv_proc = RepeatedStratifiedKFold(n_iter=repeats, n_splits=folds, random_state=seed)

    steps = list()
    for key in list(param_grid[0].keys()):
        param_grid[0]["clf__" + key] = param_grid[0].pop(key)

    if pipeline is not None:
        for step_name, step_value in pipeline:
            for step_func, step_param_grid in iter(step_value.items()):
                steps.append((step_name, step_func))
                for key in list(step_param_grid[0].keys()):
                    param_grid[0][step_name + "__" + key] = step_param_grid[0][key]

    steps.append(("clf", classifier))
    pipe = Pipeline(steps)

    gc.collect()
    start = time.time()
    _print_evaluation_header(pipe)
    logging.info("Model selection...")
    grid = skms.GridSearchCV(pipe, param_grid, cv=cv_proc, verbose=3, pre_dispatch=jobs,
                             scoring=evaluation_metric, n_jobs=jobs, iid=False)
    gc.collect()
    grid.fit(X_train, y_train)
    search_time = time.time() - start
    gc.collect()

    return grid, search_time


def train_and_test(estimator, X, y, dataset_name, preprocessing, seed, test_split, write_to_csv=True,
                   save_confusion_matrix=False, learning_curve=False, confusion_matrix=True,
                   save_predictions=True, save_model=False, save_feature_importance=True):
    """
    Performs simple classifier training and testing using a holdout procedure.
    :param estimator: classifier
    :param X: attribute values
    :param y: class values
    :param dataset_name: dataset name to distinguish different test runs in result files
    :param preprocessing: Preprocessing() object
    :param seed: random seed for data set splitting
    :param test_split: percentage of the dataset that is held out for testing
    :param write_to_csv: should the evaluation results be saved to a csv file
    :param save_confusion_matrix: should the resulting confusion matrix be saved to a file
    :param learning_curve: determines whether a learning curve will be plotted and saved to a file
    :param confusion_matrix:  determines whether the confusion matrix will be plotted and saved to a file
    :param save_predictions: saves prections (as well as probabilities and true classes) to a file
    :param save_model: saves trained classifier to a file
    :param save_feature_importance: saves feature importance to a csv and interactive html file (plot)
    """
    X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=test_split, random_state=seed, stratify=y)

    _print_evaluation_header(estimator)
    training_start = time.time()
    clf = train_classifier(estimator, X_train, y_train)
    training_time = time.time() - training_start

    testing_start = time.time()
    y_true, y_pred, y_prob, y_resolution = test_classifier(clf, X_test, y_test)
    testing_time = time.time() - testing_start
    logging.info("Evaluating")
    evaluation = Evaluation(dataset_name, preprocessing, clf, None, y_true, y_pred, y_prob, y_resolution, None, None,
                            training_time, testing_time)
    logging.info(evaluation)

    _optional_output(clf, evaluation, X, y, write_to_csv, save_confusion_matrix, learning_curve,
                     confusion_matrix, save_predictions, save_model, save_feature_importance)

    return evaluation


def run_experiment(classifiers, X, y, dataset_name, preprocessing, pipeline, seed, evaluation_metric="accuracy",
                   outer_cv=10, repeats=5, folds=2, jobs=-1, write_to_csv=True,
                   save_confusion_matrix=False, learning_curve=False, confusion_matrix=False,
                   save_predictions=False, save_model=False, save_feature_importance=False):
    """
    Runs a series of experiments selecting model parameters and testing different classifiers.
    :param classifiers:
    :param X: attribute values
    :param y: class values
    :param dataset_name: dataset name to distinguish different test runs in result files
    :param preprocessing: dataset Preprocessing() object
    :param pipeline: preprocessing pipeline steps
    :param seed: random seed for repeated stratified cross-validation
    :param test_split: percentage of the dataset that is held out for testing
    :param evaluation_metric: metric used to select best model
    :param repeats: number of repeatitions in repeated stratified cross-validation
    :param folds: number of folds in each repetition in repeated stratified cross-validation
    :param jobs: number of jobs to run in parallel.
    :param write_to_csv: should the evaluation results be saved to a csv file
    :param save_confusion_matrix: should the resulting confusion matrix be saved to a file
    :param learning_curve: determines whether a learning curve will be plotted and saved to a file
    :param confusion_matrix:  determines whether the confusion matrix will be plotted and saved to a file
    :return:
    """
    classifiers = copy.deepcopy(classifiers)

    for classifier, param_grid in iter(classifiers.items()):
        cv = skms.StratifiedKFold(n_splits=outer_cv, random_state=seed, shuffle=False)

        for fold_num, (train, test) in enumerate(cv.split(X, y)):
            clf = copy.deepcopy(classifier)
            params = copy.deepcopy(param_grid)
            pipe = copy.deepcopy(pipeline)

            logging.info("================================================================================")
            logging.info("Fold %d:", fold_num)
            logging.info("")
            X_train, X_test = X.iloc[train,], X.iloc[test,]
            y_train, y_test = y.iloc[train], y.iloc[test]

            gc.collect()
            search_results, search_time = select_model(clf, params, X_train, y_train, pipe, seed,
                                                       evaluation_metric, repeats, folds, jobs)
            gc.collect()

            testing_start = time.time()
            y_true, y_pred, y_prob, y_resolution = test_classifier(search_results, X_test, y_test)
            testing_time = time.time() - testing_start
            evaluation = Evaluation(dataset_name, preprocessing, search_results.best_estimator_, params, y_true, y_pred,
                                    y_prob, y_resolution, search_results, evaluation_metric, search_time, testing_time)
            logging.info(evaluation)

            _optional_output(search_results.best_estimator_, evaluation, X, y, write_to_csv, save_confusion_matrix,
                             learning_curve, confusion_matrix, save_predictions, save_model, save_feature_importance,
                             fold_num)
            gc.collect()


def cross_validate(classifiers, X, y, dataset_name, preprocessing, seed, cv_folds=10, write_to_csv=True,
                   save_confusion_matrix=True, learning_curve=False, confusion_matrix=False,
                   save_predictions=True, save_model=False, save_feature_importance=True):
    """
    Runs a series of experiments selecting model parameters and testing different classifiers.
    :param classifiers:
    :param X: attribute values
    :param y: class values
    :param dataset_name: dataset name to distinguish different test runs in result files
    :param preprocessing: dataset Preprocessing() object
    :param pipeline: preprocessing pipeline steps
    :param seed: random seed for repeated stratified cross-validation
    :param test_split: percentage of the dataset that is held out for testing
    :param evaluation_metric: metric used to select best model
    :param repeats: number of repeatitions in repeated stratified cross-validation
    :param folds: number of folds in each repetition in repeated stratified cross-validation
    :param jobs: number of jobs to run in parallel.
    :param write_to_csv: should the evaluation results be saved to a csv file
    :param save_confusion_matrix: should the resulting confusion matrix be saved to a file
    :param learning_curve: determines whether a learning curve will be plotted and saved to a file
    :param confusion_matrix:  determines whether the confusion matrix will be plotted and saved to a file
    :return:
    """
    classifiers = copy.deepcopy(classifiers)

    for classifier in classifiers:
        cv = skms.StratifiedKFold(n_splits=cv_folds, random_state=seed, shuffle=False)

        for fold_num, (train, test) in enumerate(cv.split(X, y)):
            clf = copy.deepcopy(classifier)

            logging.info("================================================================================")
            logging.info("Fold %d:", fold_num)
            logging.info("")
            X_train, X_test = X.iloc[train,], X.iloc[test,]
            y_train, y_test = y.iloc[train], y.iloc[test]

            gc.collect()
            training_start = time.time()
            _print_evaluation_header(clf)
            clf = train_classifier(clf, X_train, y_train)
            training_time = time.time() - training_start

            gc.collect()
            testing_start = time.time()
            y_true, y_pred, y_prob, y_resolution = test_classifier(clf, X_test, y_test)
            testing_time = time.time() - testing_start
            logging.info("Evaluating")
            evaluation = Evaluation(dataset_name, preprocessing, clf, None, y_true, y_pred, y_prob, y_resolution, None,
                                    None, training_time, testing_time)
            logging.info(evaluation)

            _optional_output(clf, evaluation, X, y, write_to_csv, save_confusion_matrix, learning_curve,
                             confusion_matrix, save_predictions, save_model, save_feature_importance, fold_num)
            gc.collect()


def compare_datasets(classifiers, data_folder, dataset_names, class_attr, selected_attr, max_num_of_classes,
                     min_examples_per_class, pipeline, seed, repeats=5, folds=2, write_to_csv=True,
                     create_box_plot=True, validation_data=None, twilight_data=None,
                     save_to_folder=os.path.join(os.path.dirname(__file__), "ExperimentResults")):
    comparison_file = "Comparison" + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".csv"
    logging.info("Finding common examples...")
    common_blobs = _get_common_blobs([os.path.join(data_folder, fix) for fix in dataset_names])

    for dataset_name in dataset_names:
        dataset_path = os.path.join(data_folder, dataset_name)
        clean_data = prep.DatasetCleaner(prep.read_dataset(dataset_path), class_attribute=class_attr,
                                         select_attributes=selected_attr, max_num_of_classes=max_num_of_classes,
                                         min_examples_per_class=min_examples_per_class, seed=seed,
                                         where_title=common_blobs, sort_by_title=True, filter_examples=True,
                                         validation_data=validation_data, twilight_data=twilight_data)

        X, y = clean_data.prepare_for_classification(class_attr, [class_attr])

        for classifier, param_grid in iter(classifiers.items()):
            steps = list()
            if pipeline is not None:
                for step_name, step_value in pipeline:
                    for step_func, step_param_grid in iter(step_value.items()):
                        steps.append((step_name, step_func))
            steps.append(("clf", classifier))
            pipe = Pipeline(steps)

            _print_evaluation_header(pipe)
            cv_proc = RepeatedStratifiedKFold(n_iter=repeats, n_splits=folds, random_state=seed)
            fold_num = 0

            for train, test in cv_proc.split(X, y):
                logging.info("Fold: " + str(fold_num))
                fold_num += 1

                training_start = time.time()
                clf = train_classifier(pipe, X.iloc[train, :], y[train])
                training_time = time.time() - training_start

                testing_start = time.time()
                y_true, y_pred, y_prob, y_resolution = test_classifier(clf, X.iloc[test, :], y[test])
                testing_time = time.time() - testing_start
                evaluation = Evaluation(dataset_name, clean_data, clf, None, y_true, y_pred, y_prob, y_resolution,
                                        None, None, training_time, testing_time)

                if write_to_csv:
                    evaluation.write_to_csv(file_name=comparison_file, save_to_folder=save_to_folder)

    if create_box_plot:
        logging.info("Plotting comparison results...")
        df = pd.read_csv(os.path.join(save_to_folder, comparison_file))
        df.loc[:, "Dataset"] = df.loc[:, "Dataset"].str.slice(0, -4)
        df.loc[df["Classifier"] == "KNeighborsClassifier", "Classifier"] = "k-NN"
        df.loc[df["Classifier"] == "RandomForestClassifier", "Classifier"] = "RF"
        df.loc[df["Classifier"] == "LogisticRegression", "Classifier"] = "Logit"
        df.loc[df["Classifier"] == "SVC", "Classifier"] = "SVM"
        df.loc[df["Classifier"] == "LGBMClassifier", "Classifier"] = "LightGBM"

        plot_comparison(df, file_name=comparison_file + ".png", save_to_folder=save_to_folder)


def lgbm_cv_early_stopping(lgbm, X, y, pipeline, seed, outer_cv=10, repeats=2, folds=5, early_stopping_rounds=10):
    cv = skms.StratifiedKFold(n_splits=outer_cv, random_state=seed, shuffle=False)

    for fold_num, (train, test) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train,], X.iloc[test,]
        y_train, y_test = y.iloc[train], y.iloc[test]

        rskf = RepeatedStratifiedKFold(random_state=seed, n_iter=repeats, n_splits=folds)
        best_iterations = []

        if pipeline is not None:
            steps = []
            for step_name, step_value in pipeline:
                for step_func, step_param_grid in iter(step_value.items()):
                    steps.append((step_name, step_func))
            pipe = Pipeline(steps)
        else:
            pipe = lgbm

        for train, test in rskf.split(X_train, y_train):
            X_train_fold = X_train.iloc[train, :].copy()
            y_train_fold = y_train.iloc[train].copy()
            X_test_fold = X_train.iloc[test, :].copy()
            y_test_fold = y_train.iloc[test].copy()

            if pipeline is not None:
                pipe_fold = clone(pipe)
                X_train_fold = pipe_fold.fit_transform(X_train_fold, y_train_fold)
                X_test_fold = pipe_fold.transform(X_test_fold)

            fit = lgbm.fit(X_train_fold, y_train_fold, early_stopping_rounds=early_stopping_rounds,
                           eval_set=[(X_test_fold, y_test_fold)], eval_metric="multi_logloss", verbose=False)
            best_iter = int(fit.best_iteration)
            best_iterations.append(best_iter)
            logging.info("Best iteration: {0} ".format(str(best_iter)))

        mean_best_iter = int(np.mean(best_iterations))
        logging.info("Fold %d: Rounded mean best number of boosting iterations: %s", fold_num, str(mean_best_iter))


def plot_comparison(df, file_name, save_to_folder=os.path.join(os.path.dirname(__file__), "ExperimentResults")):
    sns.set_style("whitegrid")

    df = util.pandas_gather(df, "Metric", "Score", ["Accuracy", "Macro recall", "Kappa", "G-mean"])

    g = sns.FacetGrid(df, row="Classifier", col="Metric", sharey=False, margin_titles=True)
    g = (g.map(sns.boxplot, "Dataset", "Score"))
    g.set_xticklabels(rotation="vertical")

    if not os.path.exists(save_to_folder):
        os.mkdir(save_to_folder)
    sns.plt.savefig(os.path.join(save_to_folder, file_name))
    sns.plt.close()


def plot_learning_curve(classifier, X, y, measurements=[0.1, 0.325, 0.55, 0.775, 1.], metric=None, n_jobs=-1,
                        save_to_folder=os.path.join(os.path.dirname(__file__), "ExperimentResults")):
    """
    Calculates the learning curve for a given model (classifier or regressor). The methods takes the evaluation
    metric as a parameter. Additionally the method saves a plot of the calculated learning curve to a file.
    :param classifier: learning model
    :type classifier: sklearn estimator
    :param X: training data
    :type X: DataFrame
    :param y: training labels
    :type y: Series
    :param measurements: number of measurements of classifier/regressor performance (number of point defining
    the learning curve)
    :type measurements: int
    :param metric: evaluation metric
    :type metric: sklearn scorer
    :param n_jobs: number of threads
    :type n_jobs: int
    :param save_to_folder: determines the folder where the learning curve plot should be saved
    :type save_to_folder: str
    :return: plt, file_name
    """
    sns.set_style("whitegrid")

    plt.figure()
    plt.title("Learning curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = skms.learning_curve(classifier, X, y, n_jobs=n_jobs,
                                                                 train_sizes=measurements, scoring=metric)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    file_name = _get_file_name("", classifier, "learning_curve", "png")
    if not os.path.exists(save_to_folder):
        os.mkdir(save_to_folder)
    plt.savefig(os.path.join(save_to_folder, file_name))
    plt.close()


def _get_common_blobs(dataset_paths, key="title", sep=";", header=0, na_values=["n/a", "nan"], low_memory=False):
    merged = pd.read_csv(dataset_paths[0], sep=sep, header=header, na_values=na_values, low_memory=low_memory)
    merged = merged.loc[merged["part_00_electrons"] > 0, :]

    for i in range(len(dataset_paths) - 1):
        df2 = pd.read_csv(dataset_paths[i + 1], sep=sep, header=header, na_values=na_values, low_memory=low_memory)
        df2 = df2.loc[df2["part_00_electrons"] > 0, :]
        merged = pd.merge(merged, df2, how="inner", on=[key])
        merged = merged.drop_duplicates(subset=key, keep="first")

    return merged[key]


def _optional_output(estimator, evaluation, X, y, write_to_csv, save_confusion_matrix,
                     learning_curve, confusion_matrix, save_predictions, save_model,
                     save_feature_importance, fold_num=None):
    if write_to_csv:
        logging.info("Saving results to file...")
        evaluation.write_to_csv(fold_num=fold_num)
    if save_confusion_matrix:
        logging.info("Saving confusion matrix to file...")
        evaluation.save_confusion_matrix()
    if learning_curve:
        logging.info("Creating learning curve...")
        plot_learning_curve(estimator, X, y)
    if confusion_matrix:
        logging.info("Plotting confusion matrix...")
        evaluation.plot_interactive_confusion_matrix()
    if save_predictions:
        logging.info("Saving predictions to file...")
        evaluation.save_predictions()
    if save_model:
        logging.info("Saving model to file...")
        evaluation.save_model()
    if save_feature_importance:
        logging.info("Saving feature importance to file...")
        evaluation.save_feature_importance()


def _print_evaluation_header(pipeline):
    logging.info("--------------------------------------------------------------------------------")
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    logging.info("")
    logging.info(str(pipeline))
    logging.info("--------------------------------------------------------------------------------")


def _get_file_name(dataset, classifier, name, file_type):
    classifier_name = type(classifier).__name__
    if classifier_name == "Pipeline":
        classifier_name = type(classifier.steps[-1][1]).__name__
        pipeline_name = classifier.steps[0][0]
    else:
        pipeline_name = "clf"

    if len(dataset) > 4 and dataset[-4] == ".":
        dataset = dataset[:-4]

    return "{0}_{1}_{2}_{3}_{4}.{5}".format(dataset, classifier_name, pipeline_name, name,
                                            time.strftime("%Y%m%d_%H%M%S", time.localtime()), file_type)