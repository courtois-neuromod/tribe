# Session Log

## 2026-02-10: Initial Setup Session

### Step 1: Preserve previous work (done)
- Created branch `previous-work` at HEAD (bf1778e + untracked CLUSTER_SETUP.md)
- Reset `main` to original initial commit `afc8d55`
- Clean working tree confirmed
- Previous work is accessible via `git log previous-work` or `git diff main..previous-work`

### Step 2: SSH multiplexing setup (done)
- Added to `~/.ssh/config` (and `~/cloud/arch_setup/home_files/all/.ssh/config` for reboot persistence):
  - `ControlMaster auto`, `ControlPath ~/.ssh/sockets/%r@%h-%p`
  - `ControlPersist yes` (persist forever)
  - `ServerAliveInterval 60` (keepalive every 60s)
- Created `~/.ssh/sockets/` directory (also added `.gitkeep` in arch_setup so it survives reboots)
- User must SSH in once manually for 2FA, then all subsequent connections reuse the socket
- **Note**: after reboot, must re-establish SSH connection (2FA) and verify `~/.ssh/sockets/` exists

### Step 3: Cluster reconnaissance (done)
- **Cluster**: Alliance Canada, `rorqual1.alliancecan.ca`, user `mleclei`
- **Working dir**: `/scratch/mleclei/` (Lustre, 19TB, use for everything)
- **Home**: 50GB quota (avoid using for large files)
- **Python**: 3.10–3.13 via `module load`
- **CUDA**: 12.2, 12.6, 12.9
- **GPUs**: H100 nodes (4 GPUs/node, 64 CPUs, 512GB RAM per node)
- **SLURM accounts**: `rrg-pbellec_gpu` for GPU jobs, `rrg-pbellec_cpu` for CPU jobs
- **GPU partitions**: `gpubase_interac` (8h), `gpubase_bynode_b1` (3h) through `b5` (7d)
- **Module load order**: `python/3.12 gcc arrow` (arrow MUST come before venv activation)
- **IMPORTANT**: Compute nodes have NO internet access. All models/packages must be pre-downloaded from login node.

### Step 4: Code fixes (done, committed)
- **Commit `2deee87`**: Fix packaging and make config cluster-agnostic
  - Added `[build-system]` to `data_utils/pyproject.toml` and `modeling_utils/pyproject.toml`
  - Replaced hardcoded paths in `defaults.py` with env var lookups (`SCRATCH`, `DATAPATH`, `SAVEPATH`, `SLURM_PARTITION`)

### Step 5: Environment setup on cluster (done, committed)
- **Commit `a762c8e`**: Add reproducible `setup_cluster.sh` and session log
- Repo cloned to `/scratch/mleclei/tribe` (reset to `afc8d55`, then synced with fixes)
- Venv created at `/scratch/mleclei/envs/tribe` with:
  - torch 2.6.0, torchvision 0.21.0, torchaudio 2.6.0
  - data_utils + modeling_utils (editable)
  - transformers, moviepy, spacy, nilearn, Levenshtein, huggingface_hub[cli], julius, h5py
- Env vars added to cluster `~/.bashrc`:
  - `DATAPATH="$SCRATCH/data"`
  - `SAVEPATH="$SCRATCH"`

### Step 6: Feature extraction scripts (done, committed)
- **Commit `086cd38`**: Add `extract_features_only.py` and `run_feature_extraction.sh`
- **Commit `9573b20`**: Rewrite with direct file logging (--logfile flag)
  - Python script: runs pipeline with `n_epochs=0`, `cluster=local` to cache features
  - SLURM script: `rrg-pbellec_gpu`, 1 GPU, 64GB, 12h, `gpubase_bynode_b3`

### Step 7: Dataset download (done)
- Dataset: `https://github.com/courtois-neuromod/algonauts_2025.competitors.git`
- Downloaded via datalad: `datalad install -r` then `datalad get -r -J8 .`
- Location: `/scratch/mleclei/algonauts_2025.competitors/` (131GB)
- **Directory structure fix**: Code expects `DATAPATH/algonauts2025/download/algonauts_2025.competitors/...`
  - Created: `mkdir -p /scratch/mleclei/data/algonauts2025/download`
  - Symlinked: `ln -sf /scratch/mleclei/algonauts_2025.competitors /scratch/mleclei/data/algonauts2025/download/algonauts_2025.competitors`

### Step 8: HuggingFace login (done)
- Logged in via `huggingface_hub.login()` (huggingface-cli not in PATH, used Python API)
- LLAMA 3.2-3B access verified with tokenizer download test

### Step 9: Pre-download models (done)
- **Compute nodes have no internet** — discovered this when spacy download hung
- Pre-downloaded from login node:
  - `en_core_web_sm` (spacy English model)
  - `meta-llama/Llama-3.2-3B` (5.7GB) — tokenizer + model
  - `facebook/w2v-bert-2.0` (2.1GB) — Wav2Vec2-BERT
  - `facebook/vjepa2-vitg-fpc64-256` (3.6GB) — VJEPA2
- All cached in `~/.cache/huggingface/hub/` on cluster

### Step 10: Setup script update (done, committed)
- Updated `setup_cluster.sh` with ALL discovered steps:
  - h5py dependency
  - Dataset directory structure + symlink
  - Model pre-downloads (spacy, LLAMA, Wav2Vec, VJEPA2)
  - Correct DATAPATH (`$SCRATCH/data` not `$SCRATCH/algonauts_2025.competitors`)

### Step 11: Test feature extraction (IN PROGRESS)
- Running on salloc job 6718981 (node rg32401, ~54 min remaining at session end)
- Script: `python -u extract_features_only.py --logfile /scratch/mleclei/feat_extract.log`
- Subset: `subject_timeline_index<10` (small test)
- Status at session end:
  - Step 1 (imports): done
  - Step 2 (init/enhancers): running `AddSentenceToWords` (spacy NLP, 0/40 timelines visible but tqdm uses \\r so file shows 0%)
  - Step 3 (feature extraction with GPU): not yet reached
- **Known issue**: tqdm writes \\r (carriage return) to log file, so progress looks stuck at 0% even when it's progressing. File size stays constant because \\r overwrites in place.
- **Job will expire** when salloc time runs out (~54 min from 21:57 UTC). May need to resubmit.

---

## Resume Checklist (next session)

### 1. Re-establish SSH connection
```bash
# After reboot, SSH multiplexing needs to be re-established
mkdir -p ~/.ssh/sockets
ssh rorqual  # Do 2FA once
```

### 2. Check if test extraction completed
```bash
ssh rorqual "cat /scratch/mleclei/feat_extract.log"
ssh rorqual "source /etc/profile; source ~/.bashrc; squeue -u mleclei"
ssh rorqual "find /scratch/mleclei/cache/algonauts-2025 -type f | wc -l"
ssh rorqual "du -sh /scratch/mleclei/cache/algonauts-2025"
```
If job expired or failed, resubmit:
```bash
ssh rorqual "source /etc/profile; source ~/.bashrc; \
  salloc --account=rrg-pbellec_gpu --partition=gpubase_interac \
  --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00 --no-shell"
# Then run:
ssh rorqual "source /etc/profile; source ~/.bashrc; \
  nohup srun --jobid=JOBID bash -c '\
  export DATAPATH=/scratch/mleclei/data SAVEPATH=/scratch/mleclei && \
  module load python/3.12 gcc arrow && \
  source /scratch/mleclei/envs/tribe/bin/activate && \
  cd /scratch/mleclei/tribe && \
  python -u extract_features_only.py --logfile /scratch/mleclei/feat_extract.log' \
  > /dev/null 2>&1 &"
```

### 3. Once test extraction succeeds
- Remove the subset query from `extract_features_only.py` (line with `subject_timeline_index<10`)
- Submit full extraction via sbatch (6-12 hours)
- Or use salloc on `gpubase_bynode_b3` (24h)

### 4. After full extraction
- Run test training: `python -m algonauts2025.grids.test_run`
- Run full grid search: `python -m algonauts2025.grids.run_grid`

### 5. Outstanding issues to fix
- **tqdm logging**: tqdm \\r overwrites don't work well in log files. Consider patching tqdm to use `\\n` instead of `\\r` when writing to files, or use `tqdm.write()` for milestones.
- **run_feature_extraction.sh**: needs `DATAPATH` and `SAVEPATH` exports added, and `h5py` is a missing dependency not in original pyproject.toml.

---

## File Locations

| What | Where |
|------|-------|
| Repo (local) | `/home/maximilienleclei/cloud/research/tribe/` |
| Repo (cluster) | `/scratch/mleclei/tribe/` |
| Venv | `/scratch/mleclei/envs/tribe/` |
| Dataset (raw) | `/scratch/mleclei/algonauts_2025.competitors/` (131GB) |
| Dataset (symlink) | `/scratch/mleclei/data/algonauts2025/download/algonauts_2025.competitors` |
| Feature cache | `/scratch/mleclei/cache/algonauts-2025/` |
| Results | `/scratch/mleclei/results/algonauts-2025/` |
| Extraction log | `/scratch/mleclei/feat_extract.log` |
| HF model cache | `~/.cache/huggingface/hub/` (on cluster) |
| Previous work | `git log previous-work` (branch on local + remote repo) |
| SSH config source | `~/cloud/arch_setup/home_files/all/.ssh/config` |

## Git History (current main)
```
9573b20 Improve feature extraction with direct file logging
11f59ad Update session log with full progress and resume checklist
a73cf90 Update setup script with dataset download and env var steps
086cd38 Add feature extraction script and SLURM job
a762c8e Add reproducible cluster setup script and session log
2deee87 Fix packaging and make config cluster-agnostic
afc8d55 Initial commit
```

## Key Reminders
- Always `module load python/3.12 gcc arrow` BEFORE activating venv on cluster
- Use `rrg-pbellec_gpu` for GPU jobs, `rrg-pbellec_cpu` for CPU jobs
- Compute nodes have NO internet — pre-download everything from login node
- SSH multiplexing: after reboot, user must `ssh rorqual` once for 2FA
- DATAPATH is `$SCRATCH/data` (not the dataset dir itself — code appends `algonauts2025`)
- Sync code: `rsync -av --exclude='.git' --exclude='logs' /home/maximilienleclei/cloud/research/tribe/ rorqual:/scratch/mleclei/tribe/`
