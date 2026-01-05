from flask import Flask, jsonify
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot Binance activo 🚀"

@app.route("/price")
def price():
    r = requests.get(
        "https://api.binance.com/api/v3/ticker/price",
        params={"symbol": "BTCUSDT"}
    )
    return jsonify(r.json())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
