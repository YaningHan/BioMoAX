# Paper result tables

Numerical tables for the paper figures, with method labels and uncertainty conventions below.

- `identity_curve_summary.csv`: Fig2A repeated-split identity curves, with named methods.
- `Fig2_print_bar_audit.csv`: Fig2B bars. Within a task/direction the order is BioMoAX, CellProfiler, DeepProfiler; expression-to-expression uses BioMoAX and L1000.
- `Fig3_print_bar_audit.csv`: Fig3 retrieval and mixing, ordered BioMoAX, CellProfiler, DeepProfiler within each group.
- `Fig4_print_bar_audit.csv`: Fig4 retrieval bars, using the same method order as Fig2, including the HT29 comparisons.
- `Fig5_print_curve_audit.csv`: named-method CRISPR retrieval curves.
- `retrieval_paired_tests.csv`: paired-test results for Fig2–4. Functional retrieval panels use the `moa` head. Holm adjustment uses the full tested comparison family, including compound-head comparisons not displayed in the figures.
- `MOA_legend_key.csv`: full mechanism/color mapping for Fig2 and Fig4. Captions show selected examples.

Bar audit values were recovered from the frozen vector figures, so small rounding differences from source statistics are expected. `low` and `high` are plotted endpoints, not a uniform confidence-interval convention: model bars show between-checkpoint SD, functional baseline bars show query-bootstrap 95% intervals, and mixing baseline bars show paired-jackknife SE. Identity-baseline uncertainty is across crop splits. Consult the paper captions before comparing intervals.

Gallery sizes depend on the task: 690 U2OS source records become 689 references when excluding a same-cell query; cross-batch queries use all 690 references. A549 same-cell retrieval excludes the query from 1,039 source records. CRISPR evaluation uses a common eligible gallery of 1,038 drugs.
