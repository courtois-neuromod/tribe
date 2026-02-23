# Current Issues & Debugging Notes

## Issue 1: tqdm progress not visible in log files
**Status**: Low priority — not the root cause of any real problem
**Symptom**: tqdm uses `\r` (carriage return), so log files look stuck at 0%.
**Workaround**: Monitor cache file count instead: `find /scratch/mleclei/cache/algonauts-2025 -type f | wc -l`

## Issue 2: Missing venv dependencies (numpy, scipy, etc.)
**Status**: RESOLVED
**Root cause**: Venv created with `include-system-site-packages=false` so system numpy/scipy/pandas were invisible. Almost every import failed silently.
**Fix**: Installed numpy, scipy, pandas, pyarrow, packaging, decorator, matplotlib, platformdirs, pygments, pillow, soundfile, mpmath into the venv. Also installed `en_core_web_lg` spacy model (code requires `lg`, only `sm` was downloaded).

## Issue 3: HuggingFace hangs on compute nodes (no internet)
**Status**: RESOLVED
**Root cause**: HuggingFace `from_pretrained()` tries to check for model updates online. Compute nodes have no internet, so it hangs indefinitely (no timeout, no error).
**Fix**: Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in environment. Added to `run_feature_extraction.sh`.

## Issue 4: run_feature_extraction.sh missing env vars and wrong resource specs
**Status**: RESOLVED
**Fix**: Added `DATAPATH`, `SAVEPATH`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` exports. Added `module load ffmpeg`. Changed GPU spec to `nvidia_h100_80gb_hbm3_1g.10gb` with 4 CPUs and 30G RAM.

## Issue 5: h5py not in original pyproject.toml
**Status**: RESOLVED
**Fix**: Added `"h5py>=3.9.0"` to `data_utils/pyproject.toml`.

## Issue 6: BFloat16 numpy conversion error
**Status**: RESOLVED
**Symptom**: `TypeError: Got unsupported ScalarType BFloat16` at `data_utils/features/text.py:255`
**Root cause**: LLAMA 3.2-3B outputs BFloat16 tensors on H100 GPUs. `.numpy()` doesn't support BFloat16.
**Fix**: Changed `word_state.cpu().numpy()` to `word_state.cpu().float().numpy()` in `text.py:255`.

## Issue 7: CVMFS cold cache causes multi-minute import hangs
**Status**: Unresolved, workaround in place
**Symptom**: `import torch` and other imports hang for 5+ minutes on some compute nodes due to cold CVMFS cache.
**Workaround**: Request specific node that has warm cache with `--nodelist=rg12502`. Not a reliable long-term fix.
**Possible fixes**:
1. Copy venv to `/localscratch` (fast local SSD) at job start
2. Use Singularity/Apptainer container
3. Pre-warm cache with dummy import step

## Issue 8: 1g.10gb GPU slice too small for LLAMA inference
**Status**: CONFIRMED — OOM error
**Error**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 174.00 MiB. GPU 0 has a total capacity of 9.75 GiB of which 150.00 MiB is free. Process has 9.59 GiB in use.`
**Details**: LLAMA 3.2-3B in FP32 (6.4GB) + batch_size=8 with all hidden states outputs = 9.6GB total. The 10GB slice is ~200MB short.
**Possible fixes**:
1. Use `2g.20gb` slice (need to fix CVMFS/torch import hang on those nodes — Issue 7)
2. Load model in FP16 (`torch_dtype=torch.float16`) — halves model size to ~3.2GB, plenty of room
3. Reduce batch_size from 8 to smaller (in `text.py:211`)
4. Use `device_map="auto"` to offload some layers to CPU
**GPU bundle options on rorqual**:
- `nvidia_h100_80gb_hbm3_1g.10gb`: 1 MIG slice, 10GB VRAM, 2 cores, 15GB RAM recommended
- `nvidia_h100_80gb_hbm3_2g.20gb`: 2 MIG slices, 20GB VRAM, 3 cores, 30GB RAM recommended
- `nvidia_h100_80gb_hbm3_3g.40gb`: 3 MIG slices, 40GB VRAM
- `nvidia_h100_80gb_hbm3_4g.40gb`: 4 MIG slices, 40GB VRAM

---

## How to check extraction status
```bash
# Job still running?
ssh rorqual "source /etc/profile; source ~/.bashrc; squeue -u mleclei"

# Cache file count (reliable progress indicator):
ssh rorqual "find /scratch/mleclei/cache/algonauts-2025 -type f | wc -l"

# GPU usage (replace JOBID):
ssh rorqual "source /etc/profile; source ~/.bashrc; srun --jobid=JOBID --overlap bash -c 'nvidia-smi'"

# Log file (may look stuck due to tqdm \r issue):
ssh rorqual "tail -5 /scratch/mleclei/tribe/feat_live.log"
```
