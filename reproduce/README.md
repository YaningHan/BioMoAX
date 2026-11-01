# Fixed-input reproduction

Run from the repository root after installing `.[reproduce]`:

```bash
python reproduce/run.py check-inputs
python reproduce/run.py statistics
python reproduce/run.py figures
python reproduce/run.py paper
```

`all` runs figures, statistics and paper compilation. SHA-256 checks protect the bundled inputs and assets before and after execution. Generated files are placed in ignored `build/` and `validation/` directories.

The package contains query-level observations, fixed UMAP coordinates, example crops and diagram assets. It verifies statistics and redraws the current five figures; it does not recreate those observations from raw images or refit UMAP. Training and encoding are separate public interfaces.

All figures use a 468-point width. UMAP markers are exported as vector paths. Installed Liberation Sans is preferred, with DejaVu Sans as fallback. Users with licensed Arial may set both `BIOMOAX_FONT_REGULAR` and `BIOMOAX_FONT_BOLD` to local font files. Complete Arial fonts are not redistributed. Numerical inputs and coordinates are fixed; different installed fonts may change glyphs and spacing.

Paper compilation requires pdfLaTeX and latexmk. It copies `paper/` into `build/paper/`, uses newly rendered figures when all five are available, and otherwise uses the approved figures included in `paper/figures/`.
