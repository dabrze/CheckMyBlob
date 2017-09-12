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

[all_summary.7z](https://onedrive.live.com/download?cid=389519B65EF435AE&resid=389519B65EF435AE%212377&authkey=AAlLFjYr9_ushHs)

The file is compressed using 7zip to allow for faster downloads. The
compressed file weighs around 1.1 GB, whereas the uncompressed CSV will
 take close to 3.0 GB of disk space. The all_summary.csv file can be
 used to reproduce the filtered ligand data sets (CMB, TAMC, CL),
 as a source to create data sets based on other filtering criteria
 (e.g. ligand subsets of your choice), or on its own a as source of
 knowledge about all ligands that CheckMyBlob was capable of detecting
 automatically on the entire PDB as of May 1st, 2017.

### CMB

CMB was designed for the CheckMyBlob study. It contains only structures
from X-ray diffraction experiments determined to at least 4.0 Å
resolution. Entries with R factor above 0.3 or
ligands below 0.3 occupancy (according to wwPDB validation reports)
were rejected. Only ligands with at least 2 non-H atoms were
considered and structures with low ligand
map correlation coefficients (RSCC < 0.6) were removed. Apart from
taking into account quality factors, we removed from the experimental
data set all moieties that are not considered proper ligands.
These included: unknown species, water molecules, standard amino acids,
and selected nucleotides. Moreover, connected ligands (as per the
naming convention in the PDB)
were labeled as alphabetically ordered strings of hetero-compound codes
(e.g., NAG-NAG-NAG-NAG). Finally, the data set was limited
to 200 most popular ligands. The resulting data set consisted of
227,885 examples with individual ligand counts ranging from
50,522 examples for SO4 (sulfate ion) to 109 for A2G
(n-acetyl-2-deoxy-2-amino-galactose). More details concerning data
selection can be found in the paper of Kowiel *et al.*


The `cmb.pkl` file is included int he repository, whereas the
csv version of the data set can be downloaded using the link below:

[cmb.csv](https://onedrive.live.com/download?cid=389519B65EF435AE&resid=389519B65EF435AE%212376&authkey=AHzE_pFDQnadMSM)

### TAMC

The TAMC data set attempts to repeat the experimental setup from
Terwilliger *et al.* described in "Ligand identification using
electron-density map correlations".
It consists of ligands from X-ray diffraction experiments with 6–150
non-H atoms. Connected PDB ligands were labeled as single
alphabetically ordered strings of hetero-compound codes, whereas
unknown species, water molecules, standard amino acids, and nucleotides
were excluded. Finally, the data set was limited to 200 most
popular ligands. The resulting data set consisted of 161,190
examples with individual ligand counts ranging from 36,466
examples for GOL (glycerol) to 114 for 3DR
(1',2'-dideoxyribofuranose-5'-phosphate).
The `tamc.pkl` file is included int he repository, whereas the
csv version of the data set can be downloaded using the link below:

[tamc.csv](https://onedrive.live.com/embed?cid=389519B65EF435AE&resid=389519B65EF435AE%212375&authkey=ANxAHbmyw7zRVrc)

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

[cl.csv](https://onedrive.live.com/embed?cid=389519B65EF435AE&resid=389519B65EF435AE%212374&authkey=AAjWc9RVe7YP5V8)

## Running the experiments

The experiments can be reproduced simply by running the
`run_experiments.py` script with appropriate parameters described below.

### Reproducing ligand data sets

To recreate the experimental data sets (CMB, TAMC, CL) the
`all_summary.csv` data set along with validation data sets (
`non_xray_pdbs.csv`, `twilight-2017-01-11`, `validation_all.csv`) have

### Evaluating selected classifiers

To evaluate the classifiers (k-NN, Random Forest, Gradient Boosting
Machines, Stacking):
```
python run_experiments.py -e
```

### Classifier tuning

Beware, this can take weeks:
```
python run_experiments.py -m
```

## Contact

If you have trouble reproducing the experiments or have any
comments/suggestions, feel free to write at
**dariusz.brzezinski (at) cs.put.poznan.pl**