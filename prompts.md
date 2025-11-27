
=====================================================================

1. Project Bootstrap & Repository Structure

=====================================================================

Create the following project layout:

earnings_nlp/
  README.md
  requirements.txt
  pyproject.toml
  config/
    config.yaml
  data_raw/
  data_processed/
  notebooks/
    01_exploration.ipynb
  src/
    __init__.py
    ingest/
      __init__.py
    preprocess/
      __init__.py
    features/
      __init__.py
    finance/
      __init__.py
    analysis/
      __init__.py
    viz/
      __init__.py
  tests/
    __init__.py


Also generate a Python script:

bootstrap_project.py
• Creates all directories.
• Creates placeholder files.
• Never overwrites existing files.
• Includes comments explaining what it does.

Also generate:

README.md

Explain in simple terms:

What earnings calls are

What Q&A section is

What stock returns, abnormal returns, and event windows mean

High-level pipeline: transcripts → sentiment → event returns → analysis

requirements.txt

Include:

pandas
numpy
matplotlib
seaborn
scikit-learn
transformers
datasets
yfinance
pyyaml
tqdm
scipy
statsmodels
streamlit

pyproject.toml

Minimal.

config/config.yaml

Include:

hf_dataset_name: "glopardo/sp500-earnings-transcripts"
sector_filter: "Health Care"
benchmark_ticker: "XBI"
price_start_date: "2013-01-01"
price_end_date: "2025-12-31"
events_base_path: "data_processed/events_base.parquet"
events_with_returns_path: "data_processed/events_with_returns.parquet"
events_with_sections_path: "data_processed/events_with_sections.parquet"
events_with_features_path: "data_processed/events_with_features.parquet"


Output all these files.

=====================================================================

2. Ingest HF Dataset & Build Events Table

=====================================================================

Generate module:

src/ingest/hf_ingest.py

With:

load_sp500_earnings_dataset(config_path: str)

filter_healthcare_calls(ds)

save_events_base(df, output_path)

main()

Use HuggingFace dataset:
glopardo/sp500-earnings-transcripts

Steps:

Load dataset

Convert to pandas

Filter to sector == "Health Care"

Keep columns:

ticker

company

sector

earnings_date

year

quarter

transcript

Convert earnings_date → datetime

Save to events_base_path

Add comments that explain:

What earnings calls are

Why we filter to Health Care

Why earnings_date is the "event date"

=====================================================================

3. Compute Stock Returns & Abnormal Returns

=====================================================================

Generate:

src/finance/returns.py

Include:

download_price_history(tickers, start, end)

compute_event_window_returns(events, prices, benchmark_ticker, window_days)

Compute:

ret_1d, ret_5d

benchmark returns

abnormal returns = stock_return − benchmark_return

Comments must explain:

What stock returns are

What benchmark ETF is

Why abnormal returns matter

What "event window" means

Generate script:

src/finance/compute_returns_for_events.py

Load base events

Download prices with yfinance

Compute returns for window_days = [1, 5]

Save to events_with_returns_path

=====================================================================

4. Parse Transcripts into Prepared Remarks vs Q&A

=====================================================================

Generate:

src/preprocess/transcript_splitter.py

Include:

normalize_transcript(text)

split_prepared_and_qa(text)

Look for Q&A markers like “Question-and-Answer Session”, “Q&A”, etc.

Fallback: detect analyst names or operator cue

add_sections_to_events(events_df)

Also generate:

src/preprocess/split_all_transcripts.py

Load events_with_returns

Apply splitting

Save to events_with_sections_path

Print 3 example splits

=====================================================================

5. FinBERT Sentiment Scoring

=====================================================================

Generate:

src/features/sentiment_finbert.py

Include:

load_finbert_pipeline()

chunk_text(text, max_tokens=256)

score_text_sentiment(text)

Score pos, neg, neu

Return dictionary with pos, neg, neu, and sentiment_score = pos − neg

add_sentiment_features(events_df)

Compute:

prep_sent_pos / neg / neu / score

qa_sent_pos / neg / neu / score

tone_shift = qa_sent_score − prep_sent_score

Generate:

src/features/compute_sentiment_features.py

Load sections

Add sentiment features

Save to events_with_features_path

=====================================================================

6. Hedging Language & Biotech Risk Terms

=====================================================================

Generate:

src/features/text_stats.py

Include:

count_terms(text, term_list)

compute_qa_text_features(df):

qa_word_count

hedging terms (may, might, uncertain, visibility…)

risk terms (FDA, trial hold, adverse event, etc.)

qa_hedge_rate

qa_risk_rate

Generate script:

src/features/compute_text_stats.py

Add new features

Save updated file

=====================================================================

7. EDA Code (Histograms, Scatterplots, Terciles)

=====================================================================

Generate:

src/analysis/eda.py

Functions:

load_events_with_features(config_path)

plot_histograms(df)

plot_scatter_sentiment_vs_returns(df)

plot_box_by_sentiment_bucket(df)

Also provide snippet for notebooks/01_exploration.ipynb.

=====================================================================

8. Statistical Tests & Simple Models

=====================================================================

Generate:

src/analysis/models.py

Include:

make_negative_positive_groups(df, feature, quantile=0.5)

compare_groups_ttest(df, feature, outcome)

run_linear_regression(df, outcome, predictors)

summarize_regression(model)

run_logistic_downdrift_model(df)

Label: abn_ret_5d < −0.05 → 1, else 0

Train/test split by time

Return metrics (accuracy, ROC-AUC, etc.)

Generate:

src/analysis/run_all_models.py

Load features

Run t-tests

Run OLS regression

Run logistic model

Print summaries

=====================================================================

9. Optional Streamlit Dashboard

=====================================================================

Generate app.py, which:

Lets user select ticker

Shows sentiment trends over time

Shows abnormal returns

Allows viewing prepared_text and qa_text for a selected call

Explains the finance terms inside the UI

=====================================================================

10. Basic Test Suite

=====================================================================

Generate pytest files:

tests/test_transcript_splitter.py

Synthetic transcripts

Test correct splitting

tests/test_returns.py

Synthetic price series

Test ret_1d and abn_ret_1d

tests/test_text_stats.py

Synthetic qa_text fields

Test hedging/risk counting

Include comment:

Run tests:
pytest -q


=====================================================================

OUTPUT FORMAT

=====================================================================

Output ALL files in order, with filenames as headers, like:

===== bootstrap_project.py =====
<content>

===== README.md =====
<content>

...and so on for every file...


Do NOT skip any file.
Do NOT shorten code.
Do NOT describe—generate full code.