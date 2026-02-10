# Session Log

## 2026-02-10: Initial Setup

### Step 1: Preserve previous work (done)
- Created branch `previous-work` at HEAD (bf1778e + untracked CLUSTER_SETUP.md)
- Reset `main` to original initial commit `afc8d55`
- Clean working tree confirmed

### Step 2: SSH multiplexing setup (done)
- Goal: persistent SSH connection to rorqual so 2FA only needed once
- Actions:
  1. Created `~/.ssh/sockets/` directory for control sockets
  2. Added to `~/.ssh/config` for rorqual host:
     - `ControlMaster auto`
     - `ControlPath ~/.ssh/sockets/%r@%h-%p`
     - `ControlPersist yes` (persist forever)
     - `ServerAliveInterval 60` (keepalive every 60s)
  3. User manually SSHed in (did 2FA)
  4. All subsequent `ssh rorqual` commands now reuse the socket
- Result: Connection verified: `rorqual1` as `mleclei`.

### Step 3: Cluster reconnaissance (done)
- **Cluster**: Alliance Canada (rorqual1.alliancecan.ca)
- **Working dir**: `/scratch/mleclei/` (faster than ~, use this for everything)
- **Scratch**: 19TB Lustre filesystem, ~19TB available, currently nearly empty
- **Home**: 50GB quota, 1.7GB used
- **Python**: 3.10, 3.11, 3.12, 3.13 available via `module load`
- **CUDA**: 12.2, 12.6, 12.9 available
- **GPUs**: H100 nodes (4 GPUs per node)
- **GPU partitions** (by time limit):
  - `gpubase_interac` — 8h (interactive)
  - `gpubase_bynode_b1` — 3h
  - `gpubase_bynode_b2` — 12h
  - `gpubase_bynode_b3` — 24h
  - `gpubase_bynode_b4` — 3 days
  - `gpubase_bynode_b5` — 7 days
  - `gpubase_bygpu_b1..b5` — same tiers, by GPU allocation
  - `gpubackfill` — opportunistic
- **Module system**: requires `source /etc/profile` in non-interactive SSH
- **Previous work**: scratch is clean (previous envs/data cleared)
- **Note**: `trash/` dirs in ~ and scratch contain undeletable files, ignore them

### Step 4: Job submission (done)
- Tested both `salloc` and `sbatch` — both work
- SLURM accounts: `rrg-pbellec_gpu` for GPU jobs, `rrg-pbellec_cpu` for CPU jobs
- GPU nodes: 64 CPUs, 512GB RAM, H100 (4/node)
- Strategy: `salloc` for interactive setup/debug, `sbatch` for long runs
- Interactive partition `gpubase_interac` max 8h
- Batch partitions up to 7 days (`gpubase_bynode_b5`)

### Step 5: Environment & code setup (done)
- Cloned repo to `/scratch/mleclei/tribe`
- Reset to initial commit `afc8d55` on cluster
- Fixed `pyproject.toml` files (added `[build-system]`) and `defaults.py` (env var lookups)
- Committed + pushed as `2deee87`
- Synced fixes to cluster via rsync
- Created venv at `/scratch/mleclei/envs/tribe`
  - **Important**: must `module load python/3.12 gcc arrow` BEFORE activating venv (Alliance Canada Arrow requirement)
  - Installed: torch 2.6.0, torchvision 0.21.0, torchaudio 2.6.0
  - Installed: data_utils + modeling_utils (editable)
  - Installed: transformers, moviepy, spacy, nilearn, Levenshtein, huggingface_hub[cli], julius
- Created `setup_cluster.sh` — reproducible setup script for anyone

### Step 6: TODO
- HuggingFace login for LLAMA 3.2 access
- Acquire Algonauts 2025 dataset
- Write feature extraction script + SLURM job
- Test training run
