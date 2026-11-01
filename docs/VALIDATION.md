# Release validation

The source release was checked on 2026-11-01 in a separate Linux directory, using the retained data and checkpoints as read-only inputs. The canonical manuscript and training datasets were not changed.

| Check | Observed result |
|---|---|
| Core, encoding and model parity tests | 9 tests passed |
| Synthetic retrieval and shell workflow | Passed, including paths with spaces, duplicate-output refusal and unknown-ID rejection |
| Fixed-input integrity | 116 bundled input/asset hashes matched |
| Numerical/statistical verification | 164 checks passed |
| Figures 1–5 | All regenerated; 468-point width; no raster images in checked UMAP regions; visual contact-sheet inspection completed |
| English manuscript | Compiled successfully, 14 total pages |
| Checkpoint integrity | All three byte counts and SHA-256 hashes matched |
| Real-input inference | Both heads of all three checkpoints matched the training model definitions exactly on the checked image/expression inputs |
| Training preflight | Two optimizer updates completed with finite losses/gradients; CRISPR-to-expression term remained zero; 90 training genes per cell |
| Actual-data record/task preparation | Expected 473/738 U2OS/A549 training records, 115 cross-source MOA queries, 32 shared-compound queries, 71 HT29 queries and 29 genetic queries recovered |
| Wheel packaging | Built successfully without downloading dependencies |

The preflight used batch size 1,000 and capped crop caches at two crops per condition. It did not run full training or save new model checkpoints. Numerical verification uses retained observations; it does not regenerate every benchmark embedding from raw data.

Observed environment: Python 3.10, PyTorch 2.1.2+cu121, NumPy 1.26.4, pandas 2.3.3, Matplotlib 3.10.8, SciPy 1.15.3 and PyMuPDF 1.28.2. GPU: NVIDIA RTX 4090. Other dependency versions and hardware may produce numerical or font-rendering differences.
