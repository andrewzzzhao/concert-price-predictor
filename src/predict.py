from datetime import datetime
import joblib
import numpy as np
import polars as pl


class ConcertPricePredictor:

  def __init__(self, artifact_path="notebooks/concert_price_model.joblib"):
    bundle = joblib.load(artifact_path)
    self.model = bundle["model"]
    self.encoder = bundle["encoder"]
    self.artist_tiers = bundle["artist_tiers"]
    self.venue_tiers = bundle["venue_tiers"]
    self.overall_median = bundle["overall_median"]

  def predict(
      self,
      artist_name: str,
      venue_name: str,
      genre: str,
      city: str,
      event_date_str: str,
  ) -> float:
    """Predicts ticket price given event details.

    event_date_str format: 'YYYY-MM-DD'
    """
    # Parse weekend indicator
    event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
    is_weekend = 1.0 if event_date.weekday() >= 5 else 0.0

    # Retrieve target-encoded tiers (fallback to global median if unseen)
    artist_match = self.artist_tiers.filter(pl.col("name") == artist_name)
    artist_tier = (
        artist_match["artist_tier_price"][0]
        if len(artist_match) > 0
        else self.overall_median
    )

    venue_match = self.venue_tiers.filter(pl.col("venue") == venue_name)
    venue_tier = (
        venue_match["venue_tier_price"][0]
        if len(venue_match) > 0
        else self.overall_median
    )

    # Historical artist count (fallback to 1 for unseen acts)
    artist_count = (
        float(artist_match["artist_event_count"][0])
        if "artist_event_count" in self.artist_tiers.columns
        and len(artist_match) > 0
        else 1.0
    )

    # One-hot encode categoricals
    cat_df = pl.DataFrame({"genre_grouped": [genre], "city_grouped": [city]})
    cat_encoded = self.encoder.transform(cat_df.to_numpy())

    # Stack all 4 continuous/numerical columns to match the 36-feature model layout
    num_features = np.array(
        [[
            np.log(artist_tier),
            np.log(venue_tier),
            np.log1p(artist_count),
            is_weekend,
        ]]
    )
    features = np.hstack([cat_encoded, num_features])

    # Predict and back-transform from log space
    log_pred = self.model.predict(features)
    return float(np.exp(log_pred)[0])


if __name__ == "__main__":
  predictor = ConcertPricePredictor()

  sample_price = predictor.predict(
      artist_name="Rumba Band",
      venue_name="Rumba Cafe",
      genre="Rock",
      city="Other",
      event_date_str="2026-10-17",
  )
  print(f"Predicted Minimum Ticket Price: ${sample_price:.2f}")