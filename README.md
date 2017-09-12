# CheckMyBlob

Machine learning experiments for the CheckMyBlob ligand identification
pipeline as described in *"Automatic recognition of ligands in electron
density by machine learning methods"* by Kowiel, M. *et al.*

## Amazon virtual machine

The easiest way to reproduce the experiments is using the publicly
available Amazon virtual machine (AMI) prepared for this study. The
machine has all the necessary libraries, scripts, and data sets
pre-installed.

To re-run the final experiments simply on the prepared AMI:
1. Request a machine on [Amazon](https://aws.amazon.com/console/)
(we used an r4.8xlarge),
2. While searching for the AMI, use the following AMI ID:
```
```
3. Log in to the instance,
4. Go to the `Classification` directory:
```
cd work/CheckMyBlob/Classification
```
5. Run the classifier evaluation:
```
python run_experiments.py -e
```

Since the evaluation will take many hours, it's best to run python
through a terminal multiplexer, like
[screen](https://help.ubuntu.com/community/Screen).

Other tasks, such as recreating data sets or parameter tuning, can also
be performed on the the machine, as described below.

## Requirements

Experiments and data sets were primarily prepared for Python 2. The code
should be compatible with Python 3, however, the `*.pkl` serialized and
compressed dataset objects have to be recreated in order to work in
Python 3 (pickle incompatibilities between python versions).

The scripts require the following libraries:
- scikit-learn
- numpy
- pandas
- scipy
- seaborn
- matplotlib
- mlxtend
- plotly
- lightgbm

## Ligand data sets

To reproduce the experiments, ligand data sets (as well as validation
files already included in the repository) should be placed in the Data
folder. Due to file size limits enforced by GitHub the ligand data sets
could not be included directly in this repository and have to be
downloaded from an external server. The descriptions and links to
the datasets are described below.

### All ligands detected in the PDB

The "master" data set containing all ligands queried and detected as described in the Kowiel et al. paper "Automatic recognition of ligands in electron density by machine learning methods" can be downloaded from:

[all_summary.7z]()

The file is compressed using 7zip to allow for faster downloads. The compressed file weighs around 1.1 GB, whereas the uncompressed CSV will take close to 3.0 GB of disk space. The all_summary.csv file can be used to reproduce the filtered ligand data sets (CMB, TAMC, CL), as source to create data sets based on other filtering criteria (e.g. ligand subsets of your choice), or on its own a as source of knowledge about all ligands to we were capable of detecting automatically on the entire PDB as of May 1st, 2017.

### CMB

[cmb.csv]()
[cmb.pkl]()

### TAMC

[tamc.csv]()
[tamc.pkl]()

### CL

[cl.csv]()
[cl.pkl]()

## Running the experimetns

The experiments can be reproduced simply by running the
`run_experiments.py` script with appropriate parameters described below.

### Reproducing ligand data sets

To recreate the experimental data sets (CMB, TAMC, CL) the
`all_summary.csv` data set along with validation data sets (
`non_xray_pdbs.csv`, `twilight-2017-01-11`, `validation_all.csv`) have

### Evaluating selected classifiers

To evaluate simple classifiers (k-NN, Random Forest, Gradient Boosting Machines):
```
python run_experiments.py -e
```



To enable easier parallelization of scripts, stacked generalization is run through a separate parameter:
```
python run_experiments.py -g
```

Th

### Classifier tuning

```
python run_experiments.py -m
```

## Contact

If you have trouble reproducing the experiments or have any comments/suggestions, feel free to write at **dariusz.brzezinski (at) cs.put.poznan.pl**