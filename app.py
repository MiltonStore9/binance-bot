# app.py

from flask import Flask
from binance.client import Client
import os

# Crear app Flask
app = Flask(__name__)

# Cliente Binance (solo precios públicos)
client = Client()  # Si quieres trading, agrega api_key y api_secret

# Ruta principal
@app.route("/")
def home():
    return "Bot Binance actighhvo 🚀"

# Ruta para precio de BTC
@app.route("/btc")
def btc():
    try:
        # Obtener precio BTC/USDT
        price = client.get_symbol_ticker(symbol="BTCUSDT")
        return f"BTCUSDT: {price['price']}"
    except Exception as e:
        return f"Error Binance: {str(e)}"

# Ejecución
if __name__ == "__main__":
    # Para Render: usar puerto asignado automáticamente
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
