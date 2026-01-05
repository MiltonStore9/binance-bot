from flask import Flask
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot Binance activo 🚀"

@app.route("/btc")
def btc():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    r = requests.get(url, timeout=5)

    if r.status_code != 200:
        return f"Error Binance: status {r.status_code}"

    try:
        data = r.json()
        return f"BTCUSDT: {data['price']}"
    except Exception as e:
        return "Binance no devolvió JSON válido"

if __name__ == "__main__":
    app.run()


