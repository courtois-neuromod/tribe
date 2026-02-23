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
- **GPUs**: H100 nodes with MIG slices (see GPU bundles in current-issues.md)
- **SLURM accounts**: `rrg-pbellec_gpu` for GPU jobs, `rrg-pbellec_cpu` for CPU jobs
- **Module load order**: `python/3.12 gcc arrow ffmpeg` (arrow MUST come before venv activation)
- **IMPORTANT**: Compute nodes have NO internet access. All models/packages must be pre-downloaded from login node.

### Steps 4-10: See previous session entries (unchanged)
- Code fixes, env setup, feature extraction scripts, dataset download, HF login, model pre-downloads, setup script

---

## 2026-02-11: Debugging & Dependency Fixes

### Diagnosis: venv was fundamentally broken
Created `diagnose.py` — a quick diagnostic script that checks all deps, spacy models, env vars, HF cache, data paths, GPU, and ffmpeg in <2 seconds.

**Root cause**: numpy was NOT installed in the venv. The venv was created with `include-system-site-packages=false` on Alliance Canada, so none of the system packages (numpy, scipy, pandas, etc.) were visible. This caused silent import failures that made everything appear "stuck".

### Fixes applied this session:

1. **Installed missing Python packages** (on cluster venv):
   - numpy, scipy, pandas, pyarrow, packaging
   - decorator, matplotlib, platformdirs, pygments, pillow
   - soundfile, mpmath
   - `en_core_web_lg` spacy model (code requires `lg`, setup only had `sm`)

2. **Added h5py to pyproject.toml**: `data_utils/pyproject.toml` now includes `h5py>=3.9.0`

3. **Fixed HuggingFace offline mode**: `from_pretrained()` was hanging because compute nodes have no internet and HF tries to phone home. Fixed by setting `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

4. **Fixed BFloat16 crash**: LLAMA 3.2-3B outputs BFloat16 on H100s, but `.numpy()` doesn't support it. Changed `word_state.cpu().numpy()` to `word_state.cpu().float().numpy()` in `data_utils/data_utils/features/text.py:255`.

5. **Updated `run_feature_extraction.sh`**:
   - Added `module load ffmpeg`
   - Added `DATAPATH`, `SAVEPATH`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` exports
   - Changed GPU spec to `nvidia_h100_80gb_hbm3_1g.10gb` with 4 CPUs, 30G RAM

6. **Updated `setup_cluster.sh`**:
   - Added numpy/scipy/pandas/pyarrow/packaging to pip install
   - Downloads `en_core_web_lg` instead of `en_core_web_sm`
   - Runs `diagnose.py` at end
   - Updated activation instructions to include `ffmpeg` module

7. **Updated `README.md`**:
   - Added Alliance Canada setup instructions (Option A with `setup_cluster.sh`)
   - Kept original conda instructions (Option B)
   - Added `diagnose.py` verification step

### Feature extraction progress:
- Enhancers (AddSentenceToWords, AddContextToWords, ExtractAudioFromVideo): **WORKING** — completes in ~30 seconds
- LLAMA model loading: **WORKING** — loads in ~80s, fits on 1g.10gb GPU (6.4GB/10GB)
- LLAMA inference: **In progress** — job running on rg12502 with BF16 fix applied. May need larger GPU slice if too slow or OOM.
- VJEPA2, Wav2VecBert: **Not yet reached**

### Outstanding issues for next session:
1. **Check if current job completed** — a feature extraction job was left running on rg12502
2. **GPU memory**: 1g.10gb slice is tight (9.1GB/10GB during LLAMA inference). May need 2g.20gb, but torch hangs on 2g.20gb slices (CVMFS issue)
3. **CVMFS cold cache**: Some nodes take 5+ min just to import torch. Workaround: `--nodelist=rg12502`
4. **VJEPA2 and Wav2VecBert**: Haven't tested these yet — may hit similar BFloat16/OOM issues

---

## Resume Checklist (next session)

### 1. Check if feature extraction job completed
```bash
ssh rorqual "source /etc/profile; source ~/.bashrc; squeue -u mleclei"
ssh rorqual "find /scratch/mleclei/cache/algonauts-2025 -type f | wc -l"
ssh rorqual "tail -20 /scratch/mleclei/tribe/feat_live.log"
```

### 2. If job failed/expired, resubmit with:
```bash
ssh rorqual "source /etc/profile && source ~/.bashrc && \
  salloc --account=rrg-pbellec_gpu \
  --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=4 --mem=30G --time=01:00:00 \
  --nodelist=rg12502 \
  srun bash -c 'module load python/3.12 gcc arrow ffmpeg && \
  source /scratch/mleclei/envs/tribe/bin/activate && \
  export DATAPATH=/scratch/mleclei/data SAVEPATH=/scratch/mleclei \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && \
  cd /scratch/mleclei/tribe && \
  python -u extract_features_only.py 2>&1 | tee feat_live.log'"
```

### 3. Sync local changes to cluster
```bash
scp /home/maximilienleclei/cloud/research/tribe/data_utils/data_utils/features/text.py rorqual:/scratch/mleclei/tribe/data_utils/data_utils/features/text.py
scp /home/maximilienleclei/cloud/research/tribe/run_feature_extraction.sh rorqual:/scratch/mleclei/tribe/run_feature_extraction.sh
```

### 4. Once feature extraction succeeds
- Run test training: `python -m algonauts2025.grids.test_run`
- Run full grid search: `python -m algonauts2025.grids.run_grid`

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
| Live log | `/scratch/mleclei/tribe/feat_live.log` |
| HF model cache | `~/.cache/huggingface/hub/` (on cluster) |
| Diagnostic script | `diagnose.py` (run on cluster to check all deps) |

## Key Reminders
- Always `module load python/3.12 gcc arrow ffmpeg` BEFORE activating venv on cluster
- Always set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` on compute nodes
- Use `rrg-pbellec_gpu` for GPU jobs, `rrg-pbellec_cpu` for CPU jobs
- Compute nodes have NO internet — pre-download everything from login node
- Prefer node `rg12502` (warm CVMFS cache) with `--nodelist=rg12502`
- DATAPATH is `$SCRATCH/data` (not the dataset dir itself — code appends `algonauts2025`)
