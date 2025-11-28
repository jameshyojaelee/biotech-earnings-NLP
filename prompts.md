### 3) Add earnings surprise / guidance proxy and include in models
```
Goal: add a simple beat/miss proxy and use it in regression/logistic models.
1) Add a function in src/finance/returns.py or new module src/finance/surprise.py to compute beat_miss_flag using price reaction as proxy: beat_miss_flag = sign(ret_1d) (1 for >=0, -1 for <0). Also allow a nullable column if future consensus data appears.
2) Modify src/finance/compute_returns_for_events.py to append beat_miss_flag after returns are computed.
3) Update src/analysis/models.py:
   - Include beat_miss_flag as a control in OLS predictors.
   - Add a second OLS model summary that shows impact of qa_sent_score/tone_shift controlling for beat_miss_flag.
4) Adjust run_all_models.py to print the new controlled regression summary.
5) Add/adjust tests to ensure beat_miss_flag is created and takes {-1, 0, 1}.
6) Run pytest -q.
Return code diffs summary and confirm tests pass.
```

### 4) Expand text features (biotech risk/hedging) with better tokenization
```
Improve domain features to catch phrase variants.
1) In src/features/text_stats.py, add simple preprocessing: lowercase, remove punctuation, and handle hyphens. Use regex tokens. Expand HEDGE_TERMS and RISK_TERMS with common biotech variants (e.g., “fda-approved”, “enrollment delays”, “data cutoff”, “adverse events”, “clinical hold”, “pdufa date”, “complete response letter”, “type c meeting”, “breakthrough designation”).
2) Support both single and multi-word term matching without strict word boundaries that drop hyphenated forms.
3) Add tests in tests/test_text_stats.py to cover hyphenated and multiword variants.
4) Run pytest -q.
Report the key additions to the lexicon and matching logic.
```

### 5) Persist analysis outputs and improve notebook
```
Create a richer, reproducible analysis artifact.
1) Expand notebooks/01_exploration.ipynb (or add notebooks/02_results.ipynb) to:
   - Load events_with_features.parquet.
   - Show summary stats of sentiment/returns.
   - Plot histograms, scatter of qa_sent_score vs abn_ret_1d/5d, boxplots by sentiment terciles.
   - Run the OLS models (with and without beat_miss_flag) and logistic model; display coefficients, p-values, and ROC-AUC.
   - Save key plots to assets/plots/*.png and tables to data_processed/*.csv (e.g., regression coefficients).
2) Add a small script src/analysis/save_figs_and_tables.py that mirrors the notebook logic to regenerate plots/tables headlessly. Provide CLI args for config paths and output dirs.
3) Update README.md “Explore” section to mention saved plots and tables locations.
4) Do not commit large outputs; just ensure code saves them when run.
Provide a brief list of generated figures/tables and how to run the script.
```

### 6) Reproducibility and data version pinning
```
Harden reproducibility.
1) In config/config.yaml add hf_dataset_revision (commit hash) and price_cache_dir (e.g., data_raw/prices). Use these in ingest and returns modules.
2) Modify src/ingest/hf_ingest.py to load the dataset with the specified revision and log it.
3) Modify src/finance/returns.py to optionally read/write cached prices per ticker to price_cache_dir before hitting yfinance; add a --refresh-cache flag to compute_returns_for_events.py.
4) Update README.md with a short “Reproducibility” note explaining dataset revision and price caching.
5) Add tests (can be lightweight/mocked) to ensure cache paths are honored (mock yfinance.download to avoid network).
6) Run pytest -q.
Summarize config changes and caching behavior.
```

### 7) Streamlit polish (optional)
```
If time allows, add a couple of polish items:
1) In app.py, add selectbox to choose return window (abn_ret_1d vs abn_ret_5d) and display chosen metric.
2) Add a scatter plot of qa_sent_score vs chosen abnormal return for the selected ticker, with a trendline.
3) Add a download button to export the ticker’s rows as CSV.
4) Upgrade the overall theme with always dark-mode theme along with more trendy colors to make it look stylish
Keep changes backward-compatible.
```