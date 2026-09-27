# Concert Minimum Ticket Price Predictor

An end-to-end machine learning pipeline built with Polars and scikit-learn to forecast face-value minimum ticket prices for live concert events.

## Architecture & Modeling
* **Data Processing:** Polars-based extraction and grouping for artist, venue, genre, and city metro distributions.
* **Target:** Natural logarithm of face-value minimum ticket price (`ln(min_price)`), converted back to real dollars at inference via exponential inversion.
* **Feature Engineering:**
  * Target-encoded artist pricing tiers (historical median entry price).
  * Target-encoded venue baseline tiers.
  * Artist tour volume / frequency weighting (`log1p(artist_event_count)`).
  * One-hot categorical encodings for grouped genre and city metro areas.
  * Weekend scheduling flags (`is_weekend`).
* **Performance:**
  * Baseline Ridge Regressor: RMSE ~$18.60 | MAE ~$9.38
  * Feature-Engineered Ridge Pipeline: **RMSE $14.24 | MAE $7.35**

## Project Structure
```text
├── notebooks/
│   ├── 01_eda.ipynb
│   └── concert_price_model.joblib
├── src/
│   └── predict.py
├── train_engineered.parquet
├── test_engineered.parquet
├── requirements.txt
└── README.md