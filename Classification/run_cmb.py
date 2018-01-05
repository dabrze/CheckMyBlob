import os
import gc
import logging
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GatherData'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

import util
import preprocessing as prep
import calculate
import postrun

SEED = 23

if __name__ == '__main__':
    pdb = sys.argv[1]
    pdb_file = sys.argv[2]
    cif_file = sys.argv[3]
    mtz_file = sys.argv[4]
    model_path = sys.argv[5]
    output_folder = sys.argv[6]

    logging.info("Starting job for {0}".format(pdb))

    ref_time, proc_time, blobs = \
        calculate.calculate(pdb, pdb_file, cif_file, mtz_file,
                            overwrite=True, logging_level=logging.WARNING, output_stats=True,
                            rerefine=False, pdb_out_data_dir=output_folder)
    examples_file = postrun.single_file(pdb)

    X = prep.DatasetCleaner(prep.read_dataset(examples_file), class_attribute=util.CLASS_ATTRIBUTE,
                            select_attributes=util.SELECTION, training_data=False).data_frame
    gc.collect()
    pred_model = util.load_model(model_path)
    logging.info("Making predictions")
    y_pred = pred_model.classifier.predict(X)
    logging.info("Calculating probabilities")
    y_proba = pred_model.classifier.predict_proba(X)
    y_pred_label = pred_model.label_encoder.inverse_transform(y_pred)
    top_10_pred = np.argsort(y_proba, axis=1)[:, -10:][:, ::-1]
    top_10_labels = pred_model.label_encoder.inverse_transform(top_10_pred)
    top_10_probabilities = np.array([subarray[index] for subarray, index in zip(y_proba, top_10_pred)])
    logging.info("Job finished")
