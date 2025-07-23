# Create test_raw_ohlc.py
import requests

url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
params = {"vs_currency": "usd", "days": 1}

response = requests.get(url, params=params)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
