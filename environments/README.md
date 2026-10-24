# Separate environments

Use separate environments for the core package and image preprocessing tools. The dependency files specify top-level packages rather than full environment lockfiles.

Core: use Python 3.10 and a machine-compatible PyTorch build (local checks used 2.1.2+cu121), then run:

```bash
python -m pip install -r environments/core.txt
python -m pip install -e .
python -m pip check
python -m unittest discover -s tests -v
```

The tested CPU installation sequence was:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-deps 'torch==2.1.2+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/core.txt -e .
python -m pip check
```

For GPU training, install a PyTorch build compatible with your CUDA environment instead of the CPU build above.

DeepProfiler: use Python 3.8 with `deepprofiler.txt`. TensorFlow 2.13 requires an older NumPy than the core package. Run the standalone pipeline without installing the core package there.

Cellpose: install `cellpose.txt` in a separate compatible PyTorch environment.

CellProfiler: `cp_wrapper.txt` installs only TIFF/CSV wrapper dependencies. The external program was tested with the existing 4.2.8.1 environment and configured Java. The file does not recreate its OS/Java/wxPython dependencies or install CellProfiler itself.

Configure `CORE_PYTHON`, `CELLPOSE_PYTHON`, `CP_PYTHON`, `DP_PYTHON` and `CP_EXECUTABLE` in your shell configuration. Run `python -m pip check` in each environment.
