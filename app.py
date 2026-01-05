from flask import Flask
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot Binance activo 🚀"

@app.route("/btc")
def btc():
    url = "https://data.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    data = requests.get(url).json()
    return f"BTCUSDT: {data['price']}"

if __name__ == "__main__":
    app.run()
