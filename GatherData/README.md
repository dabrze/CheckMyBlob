# Calculating the blob features for prediction

To detect blobs in PDB files and calculate their descriptors, follow the steps listed below.

1. Install CCP4. Due to the licensing issues, we cannot distribute CCP4 alongside our code; please
install it manually. You can get CCP4 at http://www.ccp4.ac.uk/download/index.php. In the following
steps we will denote the folder CCP4 was installed to as `$PATH_TO_CCP4 `.

1. Go to the `GatherData` directory in the CheckMyBlob repository:
```
cd GatherData
```

1. Install required packages using the following command:
```
pip install -r requirements.txt
```
Installation may take a while, because the mmLib is compiling a monomer library
during install.
 >If you skipped the above step you will have to install the mmLib library (http://pymmlib.sourceforge.net/) manually.
 >We created a fork that is working with newer numpy version, which can download it from here:
 >https://github.com/hokinus/mmLib. After downloading and extracting the library into
 >a folder we denote as `$PATH_TO_MMLIB`, run the following commands:
 >```
 >cd `$PATH_TO_MMLIB`
 >python setup.py build
 >python setup.py install
 >```
1. Compile the required libraries. You will need CCP4 to be set up correctly
and source it (we assume that you use Bash).
```
. $PATH_TO_CCP4/BINARY.setup
source $PATH_TO_CCP4/bin/ccp4.setup-sh
export LD_LIBRARY_PATH=$PATH_TO_CCP4/lib
```

1. The code in this repository was tested with CCP4 7.0 and compiled with GCC 4.8. They will not work if compiled with GCC 5.
```
# Use this if your default compiler is different than GCC 4.8
# export CC="${PWD}/lib/gcc4_compat_wrapper gcc-4.8"
cd lib/pyclipper
python setup.py install
```

1. Download the desired coordinates and structure files from PDB. You can use only part
of the repository, but the software assumes that the directory structure reflects
how data is stored by the PDB. The files should be unzipped and you will have to have
at least the following directories:
```
$PDB_REPOSITORY/structures/all/pdb
$PDB_REPOSITORY/structures/all/mmCIF
$PDB_REPOSITORY/structures/all/structure_factors
```
The naming convention of the files should follow that of the pDB archive, that is:
 - pdb*xxxx*.ent (PDB files),
 - *xxxx*.cif (mmCIF files),
 - r*xxxx*sf.ent (CIF structure factor files).

1. Edit `config.py` and provide proper paths to the files

1. If you have downloaded the files from the PDB you can convert the structure
factors in CIF format (available as rXXXXsf.ent files in the PDB archive) to MTZ format
by putting the CIF/ENT file with structure factors in the
`$PDB_REPOSITORY/structures/all/structure_factors` folder and running:
```
python cif2mtz.py
```

1. Run calculations for selected pdb files.
```
python calculate.py xxxx
```
where `xxxx` stands for the selected PDB code.

 We recommend using GNU parallel for running calculations for multiple files
```
parallel ./calculate {} < list_of_pdb_codes.txt
```

1. To convert the processed files to a format suitable for machine learning, run the following command:
```
python postrun.py
```
The data set for training classifiers will be in the folder defined by the `config.pdb_out_data_dir` parameter; this defaults to `../Output/checkmyblob/all/all_summary.txt`