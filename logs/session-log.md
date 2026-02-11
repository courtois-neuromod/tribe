# Session Log

## 2026-02-10: Initial Setup Session

### Step 1: Preserve previous work (done)
- Created branch `previous-work` at HEAD (bf1778e + untracked CLUSTER_SETUP.md)
- Reset `main` to original initial commit `afc8d55`
- Clean working tree confirmed
- Previous work is accessible via `git log previous-work` or `git diff main..previous-work`

### Step 2: SSH multiplexing setup (done)
- Added to `~/.ssh/config` for rorqual host:
  - `ControlMaster auto`, `ControlPath ~/.ssh/sockets/%r@%h-%p`
  - `ControlPersist yes` (persist forever)
  - `ServerAliveInterval 60` (keepalive every 60s)
- Created `~/.ssh/sockets/` directory
- User must SSH in once manually for 2FA, then all subsequent connections reuse the socket

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
  - transformers, moviepy, spacy, nilearn, Levenshtein, huggingface_hub[cli], julius
- Env vars added to cluster `~/.bashrc`:
  - `DATAPATH="$SCRATCH/algonauts_2025.competitors"`
  - `SAVEPATH="$SCRATCH"`

### Step 6: Feature extraction scripts (done, committed)
- **Commit `086cd38`**: Add `extract_features_only.py` and `run_feature_extraction.sh`
  - Python script: runs pipeline with `n_epochs=0`, `cluster=local` to cache features
  - SLURM script: `rrg-pbellec_gpu`, 1 GPU, 64GB, 12h, `gpubase_bynode_b3`

### Step 7: Dataset download (done)
- Dataset: `https://github.com/courtois-neuromod/algonauts_2025.competitors.git`
- Downloaded via datalad: `datalad install -r` then `datalad get -r -J8 .`
- Location: `/scratch/mleclei/algonauts_2025.competitors/` (131GB)

### Step 8: HuggingFace login (done)
- Logged in via `huggingface_hub.login()` (huggingface-cli not in PATH, used Python API)
- LLAMA 3.2-3B access verified with tokenizer download test

### Step 9: Feature extraction (TODO)
- Ready to submit — both blockers cleared (dataset + HF login)
- Start with small subset test, then full run

---

## Resume Checklist (next session)

### 1. Check dataset download
```bash
ssh rorqual "du -sh /scratch/mleclei/algonauts_2025.competitors/"
ssh rorqual "ps aux | grep datalad | grep -v grep"
ssh rorqual "tail -20 /scratch/mleclei/datalad_download.log"
```
If it failed or got killed, restart:
```bash
ssh rorqual "source /etc/profile; module load python/3.12 git-annex; export PATH=\$HOME/.local/bin:\$PATH; cd /scratch/mleclei/algonauts_2025.competitors && nohup datalad get -r -J8 . > /scratch/mleclei/datalad_download.log 2>&1 &"
```

### 2. HuggingFace login (user must do interactively)
```bash
ssh rorqual
module load python/3.12 gcc arrow
source /scratch/mleclei/envs/tribe/bin/activate
huggingface-cli login
```
- Need a `read` token from https://huggingface.co/settings/tokens
- Need access to https://huggingface.co/meta-llama/Llama-3.2-3B (request access, wait ~5-30 min)

### 3. Test feature extraction (once dataset + HF login are ready)
```bash
# Edit extract_features_only.py — uncomment the small subset query first
# Then:
cd /scratch/mleclei/tribe
sbatch run_feature_extraction.sh
squeue -u mleclei
```

### 4. Full feature extraction
- Comment out the subset query in `extract_features_only.py`
- Submit again, expect 6-12 hours

### 5. Test training run
```bash
cd /scratch/mleclei/tribe
python -m algonauts2025.grids.test_run
```

### 6. Full grid search
```bash
python -m algonauts2025.grids.run_grid
```

---

## File Locations

| What | Where |
|------|-------|
| Repo (local) | `/home/maximilienleclei/cloud/research/tribe/` |
| Repo (cluster) | `/scratch/mleclei/tribe/` |
| Venv | `/scratch/mleclei/envs/tribe/` |
| Dataset | `/scratch/mleclei/algonauts_2025.competitors/` |
| Feature cache | `$SCRATCH/cache/algonauts-2025/` (created during extraction) |
| Results | `$SCRATCH/results/algonauts-2025/` (created during training) |
| Download log | `/scratch/mleclei/datalad_download.log` |
| Previous work | `git log previous-work` (branch on local repo) |

## Git History (current main)
```
a73cf90 Update setup script with dataset download and env var steps
086cd38 Add feature extraction script and SLURM job
a762c8e Add reproducible cluster setup script and session log
2deee87 Fix packaging and make config cluster-agnostic
afc8d55 Initial commit
```

## Key Reminders
- Always `module load python/3.12 gcc arrow` BEFORE activating venv on cluster
- Use `rrg-pbellec_gpu` for GPU jobs, `rrg-pbellec_cpu` for CPU jobs
- SSH multiplexing: if connection drops, user must `ssh rorqual` once for 2FA
- Sync code to cluster: `rsync -av --exclude='.git' --exclude='logs' /home/maximilienleclei/cloud/research/tribe/ rorqual:/scratch/mleclei/tribe/`
