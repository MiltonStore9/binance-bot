from flask import Flask
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot Binance f activo 🚀"

@app.route("/btc")
def btc():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    r = requests.get(url, headers=headers, timeout=5)

    if r.status_code != 200:
        return f"Error Binance: status {r.status_code}"

    try:
        data = r.json()
        return f"BTCUSDT: {data['price']}"
    except Exception as e:
        return "Binance no devolvió JSON válido"


if __name__ == "__main__":
    app.run()




