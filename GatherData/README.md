# Calculating the blob features for prediction

To remove the non main/side-chain models from PDB files, re-refine them using REFMAC, detect and label blobs, and calculate their descriptors, follow the steps listed below.

1. Install CCP4. Due to the licensing issues, we cannot distribute CCP4 alongside our code; please
install it manually. You can get CCP4 at http://www.ccp4.ac.uk/download/index.php. In the following
steps we will denote the folder CCP4 was installed to as `$PATH_TO_CCP4 `.

2. Go to the `GatherData` directory in the CheckMyBlob repository:
```
cd GatherData
```

3. Install required packages using the following command (if you are using Anaconda
as your python distribution, you will only need the last library from the list):
```
pip install -r requirements.txt
```
Installation may take a while, because mmLib is compiling a monomer library
during installation.
 >If you skipped the above step, you will have to install the mmLib library (http://pymmlib.sourceforge.net/) manually.
 >We created a fork that is working with newer numpy versions, which  you can download from:
 >https://github.com/hokinus/mmLib. After downloading and extracting the library into
 >a folder we denote as `$PATH_TO_MMLIB`, run the following commands:
 >```
 >cd `$PATH_TO_MMLIB`
 >python setup.py build
 >python setup.py install
 >```

4. Compile the required libraries. You will need CCP4 to be set up correctly
and source it (we assume that you use Bash).
```
sh $PATH_TO_CCP4/BINARY.setup
source $PATH_TO_CCP4/bin/ccp4.setup-sh
export LD_LIBRARY_PATH=$PATH_TO_CCP4/lib
```

5. The code in this repository was tested with CCP4 7.0 and compiled with GCC 4.8. It will not work if compiled with GCC 5.
We assume you have gcc-4.8 and g++ installed.
```
# Use the line below if your default compiler is different than GCC 4.8
export CC="${PWD}/lib/gcc4_compat_wrapper gcc-4.8"
cd lib/pyclipper
python setup.py install
cd ../../
```

6. Download the desired coordinates and structure files from the PDB. You can use only part
of the repository, but the code assumes that the directory structure reflects
how data is stored by the PDB. The files should be unzipped and you will have to have
at least the following directories:
```
$PDB_REPOSITORY/structures/all/pdb
$PDB_REPOSITORY/structures/all/mmCIF
$PDB_REPOSITORY/structures/all/structure_factors
```
The naming convention of the files should follow that of the PDB archive, that is:
 - pdb*xxxx*.ent (PDB files),
 - *xxxx*.cif (mmCIF files),
 - r*xxxx*sf.ent (CIF structure factor files).

7. Edit `config.py` and provide proper paths to the above files.

8. If you have downloaded the files from the PDB you can convert the structure
factors in CIF format (available as rXXXXsf.ent files in the PDB archive) to MTZ format
by putting the files with structure factors in the
`$PDB_REPOSITORY/structures/all/structure_factors` folder and running:
```
python cif2mtz.py
```

9. Run calculations for selected pdb files.
```
python calculate.py xxxx
```
where `xxxx` stands for the selected PDB code.

We recommend using GNU parallel for running calculations for multiple files
```
parallel python calculate.py {} < list_of_pdb_codes.txt
```
where `list_of_pdb_codes.txt` is a file with several PDB codes, each in a separate line.

10. To convert the processed files to a format suitable for machine learning, run the following command:
```
python postrun.py
```
The data set for training classifiers will be in the folder defined by the `config.pdb_out_data_dir` parameter; this defaults to `../Output/checkmyblob/all/all_summary.txt`
