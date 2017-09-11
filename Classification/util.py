# coding: utf-8

import os
import pandas as pd
import numpy as np
import logging
import time

from sklearn import manifold, decomposition
from sklearn.base import _pprint
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib import offsetbox


__author__ = 'Marcin Kowiel, Dariusz Brzezinski'


class DatasetStatistics:
    """
    Dataset statistics.
    """
    def __init__(self, data_frame, class_series):
        """
        Constructor.
        :param data_frame: dataset
        :type data_frame: Pandas data frame
        :param class_attribute: class attribute column name
        :type class_attribute: string
        """
        self.examples = data_frame.shape[0]
        self.attributes = data_frame.shape[1]
        class_count = class_series.value_counts()
        self.min_examples = class_count[-1]
        self.max_examples = class_count[0]
        self.num_of_classes = class_count.__len__()
        self.classes = class_count

    def __repr__(self):
        """
        Returns a string description of a dataset containing basic statistics (number of examples, attributes, classes).
        :return: string representation
        """
        return "\texamples: {0}\r\n".format(self.examples) + \
               "\tattributes: {0}\r\n".format(self.attributes) + \
               "\tnum of classes: {0}\r\n".format(self.num_of_classes) + \
               "\tmin class examples: {0}\r\n".format(self.min_examples) + \
               "\tmax class examples: {0}\r\n".format(self.max_examples) + \
               "\tclasses: {0}".format(" ".join([str(key) + ": " + str(value)
                                                 for key, value in self.classes.iteritems()])
                                       if self.classes.shape[0] <= 200 else str(self.classes.shape[0]))


def pandas_gather(df, key, value, cols):
    """
    Utility function for transforming multiple columns in a pandas data frame into a key and value column.
    :param df: data frame to transform
    :param key: key column name
    :param value: value column name
    :param cols: columns to transform
    :return: transformed data frame
    """
    id_vars = [col for col in df.columns if col not in cols]
    id_values = cols
    var_name = key
    value_name = value
    return pd.melt(df, id_vars, id_values, var_name, value_name)


def save_model(clf, path, compress=0):
    joblib.dump(clf, path, compress=compress)


def load_model(path):
    logging.info("Loading: %s", path)
    return joblib.load(path)


def deep_repr(pipeline):
    class_name = pipeline.__class__.__name__
    return '%s(%s)' % (class_name, _pprint(pipeline.get_params(deep=True), offset=len(class_name), ),)


def classifier_repr(pipeline):
    if isinstance(pipeline, Pipeline):
        return pipeline.steps[-1][1].__class__.__name__
    else:
        return pipeline.__class__.__name__


def save_compressed_dataset(X, y, prefix="pdb", save_to_folder=os.path.join(os.path.dirname(__file__), "../Data")):
    logging.info("Compressing dataset")
    start = time.time()
    X.to_csv(os.path.join(save_to_folder, prefix + "_X.csv"), compression="gzip", index=True, header=True)
    y.to_csv(os.path.join(save_to_folder, prefix + "_y.csv"), index=True, header=True)
    logging.info("Compressed dataset in: %.2f seconds", time.time() - start)


def load_compressed_dataset(prefix="pdb", data_folder=os.path.join(os.path.dirname(__file__), "../Data")):
    logging.info("Reading: %s datasets", prefix)
    start = time.time()
    X = pd.read_csv(os.path.join(data_folder, prefix + "_X.csv"), compression="gzip", index_col=0, header=0)
    y = pd.read_csv(os.path.join(data_folder, prefix + "_y.csv"), index_col=0, header=0)
    logging.info("Read dataset in: %.2f seconds", time.time() - start)

    return X, y


def plot_embedding(data_frame, class_series, encoder, title=None, seed=23):
    X = decomposition.TruncatedSVD(n_components=2).fit_transform(data_frame)
    # tsne = manifold.TSNE(n_components=2, init="pca", random_state=seed)
    # X = tsne.fit_transform(data_frame)
    y = encoder.transform(class_series)

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    ax.set_axis_bgcolor('white')
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(class_series.iloc[i]),
                 color=plt.cm.Set1(y[i] % 10),
                 fontdict={'size': 6})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    plt.show()
