import os
import time
import requests
import polars as pl
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TICKETMASTER_API_KEY")

MAJOR_CITIES = [
    # West
    "San Francisco", "Los Angeles", "San Diego", "Seattle", "Portland", "Las Vegas", "Phoenix", "Denver",
    # Midwest
    "Chicago", "Minneapolis", "Detroit", "Columbus", "Indianapolis", "Kansas City",
    # South / Texas
    "Austin", "Dallas", "Houston", "Atlanta", "Nashville", "Miami", "Orlando", "New Orleans",
    # Northeast / Mid-Atlantic
    "New York", "Boston", "Philadelphia", "Washington", "Pittsburgh"
]

def fetch_concerts(cities=MAJOR_CITIES, pages_per_city=5):
    base_url = "https://app.ticketmaster.com/discovery/v2/events.json"
    all_records = []

    for city in cities:
        print(f"Fetching music events for {city}...")
        for page in range(pages_per_city):
            params = {
                "apikey": API_KEY,
                "classificationName": "music",
                "city": city,
                "size": 50,
                "page": page,
                "sort": "date,asc"
            }
            res = requests.get(base_url, params=params)
            
            if res.status_code == 429:
                print("Rate limit reached. Sleeping for 2 seconds...")
                time.sleep(2)
                continue
            elif res.status_code != 200:
                print(f"Warning: Failed to fetch page {page} for {city} (Status: {res.status_code})")
                continue

            events = res.json().get("_embedded", {}).get("events", [])
            for e in events:
                price_ranges = e.get("priceRanges")
                if not price_ranges or len(price_ranges) == 0:
                    continue

                price_info = price_ranges[0]
                min_p = price_info.get("min")
                max_p = price_info.get("max")
                
                if min_p is None or max_p is None:
                    continue

                venue_info = e.get("_embedded", {}).get("venues", [{}])[0]
                classifications = e.get("classifications", [{}])[0]

                all_records.append({
                    "id": e.get("id"),
                    "name": e.get("name"),
                    "city": venue_info.get("city", {}).get("name", city),
                    "state": venue_info.get("state", {}).get("stateCode"),
                    "venue": venue_info.get("name"),
                    "genre": classifications.get("genre", {}).get("name"),
                    "sub_genre": classifications.get("subGenre", {}).get("name"),
                    "event_date": e.get("dates", {}).get("start", {}).get("localDate"),
                    "min_price": float(min_p),
                    "max_price": float(max_p),
                    "currency": price_info.get("currency", "USD")
                })
            
            # Throttle to stay well under the 5 requests/sec threshold
            time.sleep(0.25)

    df = pl.DataFrame(all_records)
    # Deduplicate in case events overlap across metro boundaries
    df = df.unique(subset=["id"])
    return df

if __name__ == "__main__":
    df = fetch_concerts()
    print(f"\nSuccessfully collected {df.height} unique concerts with valid pricing!")
    print(df.head(10))

    os.makedirs("data/raw", exist_ok=True)
    df.write_parquet("data/raw/concerts_raw.parquet")
    print("\nUpdated dataset saved to data/raw/concerts_raw.parquet")