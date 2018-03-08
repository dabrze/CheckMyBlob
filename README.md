# CheckMyBlob

Machine learning experiments for the CheckMyBlob ligand identification
pipeline as described in *"Automatic recognition of ligands in electron
density by machine learning"* by Kowiel, M. *et al.*

## Amazon virtual machine

The easiest way to reproduce the experiments is using the publicly
available Amazon virtual machine (AMI) prepared for this study. The
machine has all the necessary libraries, scripts, and data sets
pre-installed.

To re-run the final experiments simply on the prepared AMI:
1. Request a machine on [Amazon](https://aws.amazon.com/console/)
(we used an r4.8xlarge),
2. While searching for the AMI, use the following AMI ID
`ami-74794f11` or lookup CheckMyBlob in the `us-east-2` (Ohio) region

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

The machine is setup to run the experiments out of the box. However,
other tasks, such as recreating data sets or parameter tuning, can also
be performed on the machine, although additional files may need
to be downloaded, as described below.

## Requirements

Experiments and data sets were primarily prepared for Python 2.7.
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

with versions listed in `requirements.txt`.

## Ligand data sets

To reproduce the experiments, ligand data sets and validation
reports should be placed in the Data folder. Due to file size limits
enforced by GitHub the ligand data sets are only included in a binary
serialized version (`*.pkl` files). CSV versions of the data sets as
 well as the "master" data set containing descriptions of all ligands
 detected in the PDB have to be downloaded from an external server.
 The descriptions and links to the data sets are described below.

### All ligands detected in the PDB

The "master" data set containing all ligands queried and detected as
described in the Kowiel et al. paper "Automatic recognition of ligands
in electron density by machine learning methods" can be downloaded from:

[all_summary.7z](https://zenodo.org/record/1040778/files/all_summary.7z) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1040778.svg)](https://doi.org/10.5281/zenodo.1040778)

The file is compressed using 7zip to allow for faster downloads. The
compressed file weighs around 1.1 GB, whereas the uncompressed CSV will
 take close to 3.0 GB of disk space. The all_summary.csv file can be
 used to reproduce the filtered ligand data sets (CMB, TAMC, CL),
 as a source to create data sets based on other filtering criteria
 (e.g. ligand subsets of your choice), or on its own a as source of
 knowledge about all ligands that CheckMyBlob was capable of detecting
 automatically on the entire PDB as of May 1st, 2017.
 
 For machine learning applications, please read the code in
 `run_experiments.py` and `preprocessing.py`, to see examples describing how
 to preprocess the data. In particular, the following attributes should not be used during model training or testing: "pdb_code", "res_id", "chain_id",
 "local_res_atom_count", "local_res_atom_non_h_count",
 "local_res_atom_non_h_occupancy_sum", "local_res_atom_non_h_electron_sum",
 "local_res_atom_non_h_electron_occupancy_sum", "local_res_atom_C_count",
 "local_res_atom_N_count", "local_res_atom_O_count",
 "local_res_atom_S_count", "dict_atom_non_h_count",
 "dict_atom_non_h_electron_sum", "dict_atom_C_count", "dict_atom_N_count",
 "dict_atom_O_count", "dict_atom_S_count", "fo_col", "fc_col", "weight_col",
 "grid_space", "solvent_radius", "solvent_opening_radius",
 "part_step_FoFc_std_min", "part_step_FoFc_std_max",
 "part_step_FoFc_std_step", "local_volume", "res_coverage", "blob_coverage",
 "blob_volume_coverage", "blob_volume_coverage_second",
 "res_volume_coverage", "res_volume_coverage_second", "skeleton_data",
 "resolution_max_limit", "part_step_FoFc_std_min", "part_step_FoFc_std_max",
 "part_step_FoFc_std_step".

The target attribute for classification is: `res_name`.

### CMB

CMB was designed for the CheckMyBlob study. It contains only structures
from X-ray diffraction experiments determined to at least 4.0 Å
resolution. Entries with R factor above 0.3 or
ligands below 0.3 occupancy (according to wwPDB validation reports)
were rejected. Only ligands with at least 2 non-H atoms were
considered and structures with low ligand
map correlation coefficients (RSCC < 0.6, RSZO <= 1, RSZD > 6.0) were removed. Apart from
taking into account quality factors, we removed from the experimental
data set all moieties that are not considered proper ligands.
These included: unknown species, water molecules, standard amino acids,
and selected nucleotides. Moreover, connected ligands (as per the
naming convention in the PDB)
were labeled as alphabetically ordered strings of hetero-compound codes
(e.g., NAG-NAG-NAG-NAG). Finally, the data set was limited
to 200 most popular ligands. The resulting data set consisted of
219,986 examples with individual ligand counts ranging from
48,490 examples for SO4 (sulfate ion) to 106 for A2G
(n-acetyl-2-deoxy-2-amino-galactose). More details concerning data
selection can be found in the paper of Kowiel *et al.*

The `cmb.pkl` file is included int he repository, whereas the
csv version of the data set can be downloaded using the link below:

[cmb.csv](https://zenodo.org/record/1191394/files/cmb.csv) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1191394.svg)](https://doi.org/10.5281/zenodo.1191394)

For machine learning (classification) purposes, the target attribute is: `res_name`.

### TAMC

The TAMC data set attempts to repeat the experimental setup from
Terwilliger *et al.* described in "Ligand identification using
electron-density map correlations".
It consists of ligands from X-ray diffraction experiments with 6–150
non-H atoms. Connected PDB ligands were labeled as single
alphabetically ordered strings of hetero-compound codes, whereas
unknown species, water molecules, standard amino acids, and nucleotides
were excluded. Finally, the data set was limited to 200 most
popular ligands. The resulting data set consisted of 161,758
examples with individual ligand counts ranging from 36,535
examples for GOL (glycerol) to 114 for LMG
(1,2-distearoyl-monogalactosyl-diglyceride).
The `tamc.pkl` file is included int he repository, whereas the
csv version of the data set can be downloaded using the link below:

[tamc.csv](https://zenodo.org/record/1040851/files/tamc.csv) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1040851.svg)](https://doi.org/10.5281/zenodo.1040851)

For machine learning (classification) purposes, the target attribute is: `res_name`.

### CL

The CL data set repeats the setup used in the study of Carolan & Lamzin
titled "Automated identification of crystallographic ligands using
sparse-density representations".
It consists of ligands from X-ray diffraction experiments
with 1.0–2.5 Å resolution. Adjacent PDB ligands were not connected.
Ligands were labeled according to the PDB naming convention.
The data set was limited to the 82 ligand types listed by Carolan &
Lamzin. The resulting data set consists of 121,360 examples with
ligand counts ranging from 42,622 examples for SO4 to 16 for
SPO (spheroidene). The `cl.pkl` file is included int he repository. The
csv version of the data set can be downloaded using the link below:

[cl.csv](https://zenodo.org/record/1040857/files/cl.csv) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1040857.svg)](https://doi.org/10.5281/zenodo.1040857)

For machine learning (classification) purposes, the target attribute is: `res_name`.

## Running the experiments

The experiments can be reproduced simply by running the
`run_experiments.py` script with appropriate parameters described below.

### Reproducing ligand data sets

To recreate the experimental data sets (CMB, TAMC, CL) the
`all_summary.csv` data set along with validation data sets (
`non_xray_pdbs.csv`, `twilight-2017-01-11`, `validation_all.csv`) have
to be present in the `CheckMyBlob/Data/` folder. Once all the source
data are available, run the following command:
```
python run_experiments.py -c
```

As a result, `*.pkl` and `*.csv` files will appear in the
`CheckMyBlob/Data/` folder.

### Evaluating selected classifiers

To evaluate the classifiers (k-NN, Random Forest, Gradient Boosting
Machines, Stacking) with pre-tuned parameters, simply run:
```
python run_experiments.py -e
```

By default, experimental results (predictions, feature importance,
confusion matrices, summaries) will appear in the
`CheckMyBlob/Classification/ExperimentResults/` folder. The main
summaries can be found in the `ExperimentResults.csv` file in the
aforementioned folder.

### Classifier tuning

To re-run parameter tuning, run the following command:
```
python run_experiments.py -m
```
Beware, this can take weeks.

Additionally, early stopping was used to determine the number of trees
to use in Gradient Boosting Machines. To re-run early stopping, type:
```
python run_experiments.py -s
```

## Detailed results

This repository contains all the experimental results reported in the
Kowiel *et al.* paper in the `CheckMyBlob/Results/` folder. The source
of Table 1 can be found in `Summary.xlsx`.

## Reproducing data figures

To reproduce data figures from the Kowiel *et al.* paper, go to the
`CheckMyBlob/Figures/` folder and run the following commands:

```
python accuracy.py
python labels.py
python plot_importances.py
```

For the scripts to work, the detailed experimental results have to be
present in the `CheckMyBlob/Results/` folder. Therefore, if you want to
reproduce the figures on the Amazon AMI, you will have to upload the
experimental results to the machine (the machine was prepared to re-run
experiments and does not contain any result files to avoid confusion).
The results can be easily uploaded to the AMI by updating the
repository on the machine by running `git pull` in the `CheckMyBlob/`
folder.

Figures depicting ligands can be reproduced using
[PyMOL](https://pymol.org/). The data and scripts required to 
reproduce these images are available in the 
`CheckMyBlob/Figures/pymol` directory.

## Reproducing time measurement

The maps and ARP/PHENIX ligand definitions used to measure ligand 
identification time can be found in the 
`CheckMyBlob/Data/TimeExperiments` folder. The experimental script
is in the `CheckMyBlob/Classification` and can be run as follows:

```
python time_experiments.py
```

Prior to running the scripts, PHENIX and ARP/WARP have to be installed
and the CheckMyBlob blob identification pipeline has to be set up, as
described in the `CheckMyBlob/GatherData` folder.

## Contact

If you have trouble reproducing the experiments or have any
comments/suggestions, feel free to write at
**dariusz.brzezinski (at) cs.put.poznan.pl**