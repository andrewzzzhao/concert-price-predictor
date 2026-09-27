from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent


class ConcertPricePredictor:

    def __init__(
        self,
        artifact_path=BASE_DIR / "notebooks" / "concert_price_model.joblib",
        data_path=BASE_DIR / "train_engineered.parquet",
    ):
        bundle = joblib.load(artifact_path)
        self.model = bundle["model"]
        self.encoder = bundle["encoder"]
        self.artist_tiers = bundle["artist_tiers"]
        self.venue_tiers = bundle["venue_tiers"]
        self.overall_median = bundle["overall_median"]

        self.reference_df = pl.read_parquet(data_path)

    def predict(
        self,
        artist_name: str,
        venue_name: str,
        genre: str,
        city: str,
        event_date_str: str,
    ) -> float:
        """Predicts minimum ticket price given event details."""
        event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
        is_weekend = 1.0 if event_date.weekday() >= 5 else 0.0

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

        artist_count = (
            float(artist_match["artist_event_count"][0])
            if "artist_event_count" in self.artist_tiers.columns
            and len(artist_match) > 0
            else 1.0
        )

        cat_df = pl.DataFrame({"genre_grouped": [genre], "city_grouped": [city]})
        cat_encoded = self.encoder.transform(cat_df.to_numpy())

        num_features = np.array(
            [[
                np.log(artist_tier),
                np.log(venue_tier),
                np.log1p(artist_count),
                is_weekend,
            ]]
        )
        features = np.hstack([cat_encoded, num_features])

        log_pred = self.model.predict(features)
        return float(np.exp(log_pred)[0])

    def search_and_predict(self, query: str, event_date_str: str) -> dict:
        """Finds matching concerts by keyword and predicts ticket price."""
        matches = self.reference_df.filter(
            pl.col("name").str.to_lowercase().str.contains(query.lower())
            | pl.col("venue").str.to_lowercase().str.contains(query.lower())
        )

        if len(matches) == 0:
            print(f"No concerts found matching '{query}'. Using baseline defaults.")
            artist = query
            venue = "Other"
            genre = "Other"
            city = "Other"
        else:
            top_match = matches.head(1)
            artist = top_match["name"][0]
            venue = top_match["venue"][0]
            genre = (
                top_match["genre_grouped"][0]
                if "genre_grouped" in top_match.columns
                else "Other"
            )
            city = (
                top_match["city_grouped"][0]
                if "city_grouped" in top_match.columns
                else "Other"
            )
            print(
                f"Matched: '{artist}' at '{venue}' (Genre: {genre}, City: {city})"
            )

        predicted_price = self.predict(
            artist_name=artist,
            venue_name=venue,
            genre=genre,
            city=city,
            event_date_str=event_date_str,
        )

        return {
            "query": query,
            "artist": artist,
            "venue": venue,
            "genre": genre,
            "city": city,
            "date": event_date_str,
            "predicted_price": round(predicted_price, 2),
        }


if __name__ == "__main__":
    predictor = ConcertPricePredictor()

    search_input = input("Enter artist or venue to search: ").strip()
    date_input = input("Enter target date (YYYY-MM-DD) [e.g. 2026-10-24]: ").strip()
    if not date_input:
        date_input = "2026-10-24"

    result = predictor.search_and_predict(search_input, date_input)
    print(f"\nEstimated Minimum Entry Price: ${result['predicted_price']:.2f}")