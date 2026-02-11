# Current Issues & Debugging Notes

## Issue 1: tqdm progress not visible in log files
**Status**: Unresolved, workaround needed
**Symptom**: `AddSentenceToWords` enhancer shows `0%|          | 0/40` and never updates in the log file.
**Root cause**: tqdm writes progress using `\r` (carriage return) which overwrites the same line. In a file, this means the byte offset stays the same — the file appears static even though data is being written.
**Why stdbuf/PYTHONUNBUFFERED don't help**: tqdm bypasses libc buffering and writes directly to the file descriptor. `\r` without `\n` means the file doesn't grow.
**Possible fixes**:
1. Set `TQDM_MININTERVAL` env var to a large value to reduce writes
2. Patch tqdm to use `\n` instead of `\r` when output is not a TTY:
   ```python
   import tqdm
   # Force newline mode for non-TTY outputs
   tqdm.tqdm = functools.partial(tqdm.tqdm, ascii=True, mininterval=30)
   ```
3. Use `tqdm.write()` calls at milestones instead of the progress bar
4. Monitor progress by checking cache file count instead of log:
   ```bash
   watch -n 10 'find /scratch/mleclei/cache/algonauts-2025 -type f | wc -l'
   ```

## Issue 2: AddSentenceToWords enhancer is slow
**Status**: Unknown if this is a real problem or just looks stuck due to Issue 1
**Symptom**: The spacy `AddSentenceToWords` enhancer has been running for 5+ minutes on 40 timelines with no visible progress.
**Possible causes**:
1. Spacy `en_core_web_sm` is slow on long transcripts (Friends episodes have lots of dialogue)
2. The `max_unmatched_ratio: 0.05` setting may cause retries
3. It's actually progressing fine but we can't see it (Issue 1)
**How to verify**: Check cache file count growth, or check CPU usage of the python process
**Next step**: Let it run and check results when salloc expires. If it completed, the enhancer is just slow. If not, investigate.

## Issue 3: Feature extraction hasn't reached GPU phase yet
**Status**: Blocked by Issue 2
**Symptom**: After 5+ minutes, still in CPU-only enhancer phase. GPU shows 0% utilization, 0MiB memory.
**Expected flow**:
1. Study loading (cached, instant) ✅
2. Enhancers (AddText, AddSentenceToWords, AddContextToWords, RemoveMissing, ExtractAudioFromVideo, ChunkEvents) ← STUCK HERE
3. Feature preparation (LLAMA3.2, VJEPA2, Wav2VecBert) — this is the GPU phase
4. DataLoader construction
**Next step**: Wait for enhancers to finish, then verify GPU is utilized during feature extraction.

## Issue 4: run_feature_extraction.sh needs env vars
**Status**: Not yet fixed
**Details**: The SLURM batch script doesn't export `DATAPATH` and `SAVEPATH`. If submitting via sbatch instead of salloc, these need to be set:
```bash
# Add to run_feature_extraction.sh:
export DATAPATH=/scratch/mleclei/data
export SAVEPATH=/scratch/mleclei
```

## Issue 5: h5py not in original pyproject.toml
**Status**: Workaround applied (installed manually), not fixed in pyproject.toml
**Details**: `data_utils/data_utils/studies/algonauts2025.py` imports h5py but it's not listed in `data_utils/pyproject.toml`. We installed it manually with `pip install h5py`.
**Fix**: Add `"h5py>=3.0"` to `data_utils/pyproject.toml` dependencies.

---

## How to check extraction status
```bash
# Log file (may look stuck due to tqdm \r issue):
ssh rorqual "cat /scratch/mleclei/feat_extract.log"

# Cache file count (reliable progress indicator):
ssh rorqual "find /scratch/mleclei/cache/algonauts-2025 -type f | wc -l"

# Cache size:
ssh rorqual "du -sh /scratch/mleclei/cache/algonauts-2025"

# Process alive?
ssh rorqual "source /etc/profile; source ~/.bashrc; srun --jobid=JOBID --overlap bash -c 'ps aux | grep python | grep -v grep'"

# GPU usage:
ssh rorqual "source /etc/profile; source ~/.bashrc; srun --jobid=JOBID --overlap bash -c 'nvidia-smi'"

# Job still running?
ssh rorqual "source /etc/profile; source ~/.bashrc; squeue -u mleclei"
```
