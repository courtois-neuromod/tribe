# DME: Deep multimodal brain encoding
This repository contains the Brain & AI team's submission for the Algonauts 2025 competition. It can be used for training and evaluating encoding models to predict fMRI brain responses to naturalistic video stimuli.

## Create the environment

### Option A: Alliance Canada cluster (recommended)

Run the setup script from a login node:

```bash
bash setup_cluster.sh
```

This handles everything: venv creation, dependency installation, dataset download, model pre-downloading, and environment variables. See the script for details.

To activate the environment in future sessions or SLURM jobs:

```bash
module load python/3.12 gcc arrow ffmpeg
source $SCRATCH/tribe/envs/tribe/bin/activate
export DATAPATH="$SCRATCH/data"
export SAVEPATH="$SCRATCH"
```

### Option B: Conda (local / other clusters)

**1.** Create a conda environment:

```bash
export ENVNAME=algonauts-2025
conda create -n $ENVNAME python=3.12 ipython -y
conda activate $ENVNAME

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

git clone https://github.com/courtois-neuromod/tribe.git
cd tribe/data_utils
pip install -e .
cd ../modeling_utils
pip install -e .

pip install numpy scipy pandas pyarrow packaging
pip install transformers moviepy spacy nilearn Levenshtein "huggingface_hub[cli]" julius h5py
pip install decorator matplotlib platformdirs pygments pillow
```

**2.** Download the spacy English model (`en_core_web_lg`, not `sm`):

```bash
python -m spacy download en_core_web_lg
```

**3.** Get access to the [LLAMA3.2-3B repository on HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-3B). Run:

```bash
huggingface-cli login
```

Then, create a `read` [token](https://huggingface.co/settings/tokens) and copy when prompted.

**4.** Set paths to the Algonauts dataset, where you want to save your results, and what Slurm partition to use. This can be done by setting corresponding values in `algonauts2025/grids/defaults.py`, or alternatively, by adding the following to your shell's startup file (e.g., `.bashrc`, `.zshrc`, etc.).

```bash
export SAVEPATH="/your/save/directory"
export DATAPATH="/path/to/algonauts/dataset"
export SLURM_PARTITION="your-slurm-partition"
```

**5.** Verify your setup:

```bash
python diagnose.py
```

## Run a test training locally

```
python -m algonauts2025.grids.test_run
```

## Run a grid search on Slurm

```
python -m algonauts2025.grids.run_grid
```

## Train an ensemble of models

```
python -m algonauts2025.grids.run_ensemble
```

Training and results can be monitored using [Weights & Biases](https://docs.wandb.ai/quickstart). See the config key `wandb_config`.


## License

This repository is CC-BY-NC licensed, as found in the LICENSE file. Also check-out Meta Open Source [Terms of Use](https://opensource.fb.com/legal/terms/) and [Privacy Policy](https://opensource.fb.com/legal/privacy/).
