# Metric Ion Classification Tutorial

## Installation

To install MIC:

```
git clone https://github.com/keiserlab/metric-ion-classification.git 
cd metric-ion-classification
conda env create --name mic-env -f mic-env.yml
conda activate mic-env
pip install .
```    

Installation may take some time, especially regarding conda dependency solving. We strongly recommend using an updated conda with conda-libmamba-solver, in which case installation should take a few minutes.

If you encounter issues intalling LUNA, see https://github.com/keiserlab/LUNA installation instructions.

### Requirements

The dependencies for MIC are listed in mic-env.yml, reproduced here:
```
  - rdkit
  - seaborn
  - openbabel
  - pip
  - pymol-open-source
  - xopen
  - colorlog
  - networkx
  - scipy
  - pytorch
  - cpuonly
  - numpy=1.21.6
  - biopython==1.72
  - mmh3==2.5.1
  - pdbecif
  - joblib
  - tqdm
  - scikit-learn
  - simplejson
```

### System requirements:
* MIC has been successfully installed and tested on Ubuntu 18.04, 20.04, and 22.04, Mac OS Monterey through Sonoma, and on a Windows machine running Ubuntu through WSL.

## Running MIC

### Command line

To run:
```
mic_predict -o <outfile> -e <path/to/entry_file>[optional] </path/to/pdb_file>
```
Inputs
* pdb\_file / pdb\_dir : path to either 1) a single PDB file in the case of running one 1 file or 2) a directory containing multuple PDB files, used with --entries flag 

Options:
* -e/--entries: Run individual examples from a single file or multiple PDB files. See below.
* -n/--num\_processes: Number of concurrent processes to use for FP generation - default 8
* -o/--outfile: CSV created to store generated fingerprints or MIC predictions. If not provided, output is displayed in the terminal.
* -p/--preds\_only: Only output predictions, not full probabilities from SVC.
* -ext/--extended\_set: Generate predictions for the extended set of labels (K, Mn,Iod,Fe,Br)
* -cpu/--cpu\_only: Ignore available GPUs and generate all predictions on CPU.
* -co/--symexp\_cutoff: Radius in angstroms around densities kept following symmetry expansion. If running a single PDB file, it is recommended to use 7 for X-ray crystallography or -1 for Cryo-EM structures. Only use other values if you are planning on training a new model - default 7

The following options are NOT recommended to change from their defaults, unless intending to train a new model. Do not change these if running predictions alone.
* -t/--fp\_type: Type of fingerprint to generate and use, options are ['prune-eifp']
* -l/--length: length of the fingerprint vector to be generated - default 4096
* -sn/--shell\_number: number of shells to use during fingerprint generation - default 18
* -sr/--shell\_radius: shell radius to use during fingerprint generation - default 0.25
* -ub/--unblinded: flag to not zero the invariants of spherical densities
* -b/--bit: whether to generate count fingerprints or bit fingerprints - default is count 

#### Entries

Entries are used to generate predictions for specific densities within a single PDB file or across multiple PDB files in one MIC run. To run with entries, the first argument to mic\_predict should be a path to the directory containing all pdb files, while the entries argument should be a csv in the format 'pdb\_id,chain,name,res\_number,skip\_symexpansion (must be on for Cryo-EM structures)':

```
4OKE,A,MG,202,
3A09,A,CA,601,
4L9P,B,ZN,601,
7aue,R,CA,501,X
7f53,R,CA,601,X
7f55,R,CA,601,X
7f54,R,CA,601,X
7f58,R,CA,602,X
7piu,R,CA,401,X
7piv,R,CA,401,X
```

## Testing

To confirm correct installation:

```
pip install pytest
python -m pytest
```

The tests may take a 5-10 minutes to run.

## ChimeraX Bundle

Actively under development, coming soon!

### Citing MIC

If you use this software, please cite our [preprint](https://www.biorxiv.org/content/10.1101/2024.03.18.585639v1).

