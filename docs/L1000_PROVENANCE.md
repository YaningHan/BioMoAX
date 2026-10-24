# L1000 expression inputs

Use `level3_beta_trt_cp_n1805898x12328.gctx` with `instinfo_beta.txt` from the [data provider](DATA_SOURCES.md). Level-5 signatures are not interchangeable with these Level-3 inputs.

Run `scripts/08_extract_expression.sh` with a cell name and a CSV containing `record_id,broad_id`; the full command is in [the pipeline guide](../pipelines/README.md#l1000-expression). The extractor averages same-cell compound instances without QC, dose, or time filtering. It preserves input record order and writes gene IDs, contributing sample IDs, and availability flags alongside the expression matrix. Missing profiles are marked NaN.

`manifests/drug_records.csv` describes the processed row mappings used by the paper tasks. Check record alignment, gene order, coverage and numerical compatibility when substituting newly extracted arrays for existing inputs.
