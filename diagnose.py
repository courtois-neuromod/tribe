#!/usr/bin/env python
"""Quick diagnostic: check all deps and configs before running feature extraction."""
import sys
import importlib

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def ok(msg):
    print(f"  {GREEN}OK{RESET}  {msg}")

def fail(msg):
    print(f"  {RED}FAIL{RESET}  {msg}")

def warn(msg):
    print(f"  {YELLOW}WARN{RESET}  {msg}")

errors = 0

# 1. Check critical Python packages
print("\n=== 1. Python packages ===")
for pkg in ["h5py", "Levenshtein", "spacy", "moviepy", "transformers",
            "torch", "tqdm", "exca", "nibabel", "julius"]:
    try:
        importlib.import_module(pkg)
        ok(pkg)
    except ImportError as e:
        fail(f"{pkg}: {e}")
        errors += 1

# 2. Check spacy models
print("\n=== 2. Spacy models ===")
try:
    import spacy
    import spacy.util
    for model in ["en_core_web_sm", "en_core_web_lg"]:
        if spacy.util.is_package(model):
            ok(f"{model} installed")
        else:
            if model == "en_core_web_lg":
                fail(f"{model} NOT installed — code requires this! (defaults.py maps english -> en_core_web_lg)")
                errors += 1
            else:
                warn(f"{model} not installed (not critical if en_core_web_lg is present)")
except Exception as e:
    fail(f"spacy check failed: {e}")
    errors += 1

# 3. Check what model the code actually requests
print("\n=== 3. Code config check ===")
try:
    from data_utils.utils import get_spacy_model
    import inspect
    src = inspect.getsource(get_spacy_model)
    if "en_core_web_lg" in src:
        print(f"  INFO  Code requests en_core_web_lg for English")
    elif "en_core_web_sm" in src:
        print(f"  INFO  Code requests en_core_web_sm for English")
except Exception as e:
    warn(f"Could not inspect get_spacy_model: {e}")

# 4. Check environment variables
print("\n=== 4. Environment variables ===")
import os
for var in ["DATAPATH", "SAVEPATH", "SCRATCH", "HF_HOME", "TRANSFORMERS_CACHE"]:
    val = os.environ.get(var)
    if val:
        ok(f"{var}={val}")
    else:
        if var in ("DATAPATH", "SAVEPATH"):
            fail(f"{var} not set!")
            errors += 1
        else:
            warn(f"{var} not set")

# 5. Check HuggingFace models cached locally
print("\n=== 5. HuggingFace model cache ===")
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
if os.path.isdir(hf_home):
    models = [d for d in os.listdir(hf_home) if d.startswith("models--")]
    if models:
        for m in sorted(models):
            ok(m.replace("models--", "").replace("--", "/"))
    else:
        warn(f"No models found in {hf_home}")
else:
    warn(f"HF cache dir not found: {hf_home}")

# 6. Check data paths
print("\n=== 6. Data paths ===")
datapath = os.environ.get("DATAPATH", "")
savepath = os.environ.get("SAVEPATH", "")
paths_to_check = {
    "dataset": f"{datapath}/algonauts2025/download/algonauts_2025.competitors",
    "cache": f"{savepath}/cache/algonauts-2025",
}
for name, path in paths_to_check.items():
    if path and os.path.exists(path):
        ok(f"{name}: {path}")
    elif path:
        warn(f"{name}: {path} does not exist")
    else:
        fail(f"{name}: path not configured")

# 7. Check GPU
print("\n=== 7. GPU ===")
try:
    import torch
    if torch.cuda.is_available():
        ok(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        warn("No CUDA GPU available (OK if running on login node)")
except Exception as e:
    warn(f"GPU check: {e}")

# 8. Check ffmpeg (needed by moviepy for audio extraction)
print("\n=== 8. ffmpeg ===")
import shutil
if shutil.which("ffmpeg"):
    ok("ffmpeg found in PATH")
else:
    fail("ffmpeg not in PATH — needed by moviepy for ExtractAudioFromVideo")
    errors += 1

# Summary
print(f"\n{'='*50}")
if errors:
    print(f"{RED}{errors} issue(s) found — fix these before running feature extraction{RESET}")
else:
    print(f"{GREEN}All checks passed!{RESET}")
sys.exit(errors)
