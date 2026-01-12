import time
import requests
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from flask import Flask, request, render_template_string
import plotly.graph_objects as go

app = Flask(__name__)

# =========================================================
# CONFIG
# =========================================================
@dataclass
class GridConfig:
    symbol: str = "BTCUSDT"
    fee_rate: float = 0.001
    quote_start: float = 1000.0
    order_quote: float = 25.0
    grids: list = None
    start_price: float = None
    allow_multi_fills_per_bar: bool = True
    slippage: float = 0.0
    show_timezone: str = "America/Lima"

    init_mode: str = "balanced"
    base_start_custom: float = 0.0
    quote_start_custom: float = 1000.0


# =========================================================
# BINANCE: velas 1m EN MEMORIA
# =========================================================
BINANCE_SPOT = "https://api.binance.com"

def ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def fetch_klines_1m(symbol: str, start_utc: datetime, end_utc: datetime, limit: int = 1000) -> pd.DataFrame:
    start_ms = ms(start_utc)
    end_ms = ms(end_utc)

    rows, cur = [], start_ms

    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": cur,
            "endTime": end_ms,
            "limit": limit
        }
        r = requests.get(BINANCE_SPOT + "/api/v3/klines", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        rows.extend(data)
        last_open_time = data[-1][0]
        next_cur = last_open_time + 60_000
        if next_cur <= cur:
            break
        cur = next_cur

        time.sleep(0.08)
        if len(data) < limit:
            break

    if not rows:
        raise RuntimeError("No se recibieron klines en ese rango. Revisa symbol/fechas.")

    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)

    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("time").sort_index()
    return df[["open","high","low","close","volume"]]


# =========================================================
# HELPERS
# =========================================================
def classify_levels(grids_sorted, start_price):
    sells = [g for g in grids_sorted if g > start_price]
    buys  = [g for g in grids_sorted if g < start_price]
    return sells, buys

def compute_midline(grids_sorted, start_price):
    sells, buys = classify_levels(grids_sorted, start_price)
    if not sells or not buys:
        return None, None, None
    lowest_sell = min(sells)
    highest_buy = max(buys)
    midline = (lowest_sell + highest_buy) / 2.0
    return midline, lowest_sell, highest_buy

def init_balanced_inventory(cfg: GridConfig, grids_sorted):
    if cfg.start_price is None:
        raise ValueError("cfg.start_price es obligatorio.")

    total = float(cfg.quote_start)

    if cfg.init_mode == "all_usdt":
        return total, 0.0

    if cfg.init_mode == "custom":
        return float(cfg.quote_start_custom), float(cfg.base_start_custom)

    sells, buys = classify_levels(grids_sorted, cfg.start_price)
    n_sells, n_buys = len(sells), len(buys)

    if n_sells == 0 and n_buys == 0:
        usdt_for_buys = total * 0.5
    else:
        usdt_for_buys = total * (n_buys / max(1, (n_buys + n_sells)))

    usdt_for_sells = total - usdt_for_buys
    btc_for_sells = usdt_for_sells / float(cfg.start_price)
    return usdt_for_buys, btc_for_sells


# =========================================================
# CROSS DETECTION (MECHAS HIGH/LOW)
# =========================================================
def crossed_down_hilo(prev_close: float, lo: float, level: float) -> bool:
    return (prev_close > level) and (lo <= level)

def crossed_up_hilo(prev_close: float, hi: float, level: float) -> bool:
    return (prev_close < level) and (hi >= level)


# =========================================================
# SIMULADOR
# =========================================================
def simulate_grid_hilo(df: pd.DataFrame, cfg: GridConfig):
    if cfg.grids is None or len(cfg.grids) < 2:
        raise ValueError("cfg.grids debe ser una lista de >=2 niveles")
    if cfg.start_price is None:
        raise ValueError("cfg.start_price es obligatorio")

    grids_real = sorted([float(x) for x in cfg.grids])
    start_price = float(cfg.start_price)

    midline, lowest_sell, highest_buy = compute_midline(grids_real, start_price)

    ladder = grids_real.copy()
    if midline is not None:
        if all(abs(midline - g) > 1e-9 for g in ladder):
            ladder.append(midline)
        ladder = sorted(ladder)
    else:
        ladder = sorted(ladder)

    idx = {g: i for i, g in enumerate(ladder)}

    quote, base = init_balanced_inventory(cfg, grids_real)
    quote0, base0 = quote, base

    long_open = {}
    short_open = {}
    events, pairs = [], []

    def exec_price(px, side):
        if cfg.slippage <= 0:
            return px
        return px * (1 + cfg.slippage) if side == "BUY" else px * (1 - cfg.slippage)

    prev_close = float(df["close"].iloc[0])

    for k in range(1, len(df)):
        row = df.iloc[k]
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])
        ts = df.index[k]

        down_crossed = [g for g in ladder if crossed_down_hilo(prev_close, lo, g)]
        up_crossed   = [g for g in ladder if crossed_up_hilo(prev_close, hi, g)]

        if not cfg.allow_multi_fills_per_bar:
            if down_crossed:
                down_crossed = [down_crossed[0]]
            if up_crossed:
                up_crossed = [up_crossed[0]]

        # CIERRES
        for g in up_crossed:
            to_close = [buy_level for buy_level, info in long_open.items() if info["target_level"] == g]
            for buy_level in to_close:
                info = long_open[buy_level]
                qty = info["qty"]
                if base + 1e-12 < qty:
                    continue

                px = exec_price(g, "SELL")
                proceeds = qty * px
                fee = proceeds * cfg.fee_rate
                net = proceeds - fee

                base -= qty
                quote += net

                buy_cost = qty * info["buy_price"] + info["fee_buy"]
                pnl = net - buy_cost

                pairs.append({
                    "type": "BUY->SELL",
                    "open_time": info["time_buy"],
                    "close_time": ts,
                    "open_price": info["buy_price"],
                    "close_price": px,
                    "open_grid": info["buy_grid"],
                    "close_grid": g,
                    "qty": qty,
                    "fee_open": info["fee_buy"],
                    "fee_close": fee,
                    "pnl": pnl
                })

                events.append({
                    "time": ts, "side": "SELL", "price": px, "grid": g,
                    "qty": qty, "fee_quote": fee,
                    "quote_after": quote, "base_after": base,
                    "kind": "LONG_CLOSE", "pnl_pair": pnl
                })

                del long_open[buy_level]

        for g in down_crossed:
            to_close = [sell_level for sell_level, info in short_open.items() if info["target_level"] == g]
            for sell_level in to_close:
                info = short_open[sell_level]
                qty = info["qty"]

                px = exec_price(g, "BUY")
                cost = qty * px
                fee = cost * cfg.fee_rate
                total = cost + fee

                if quote + 1e-12 < total:
                    continue

                quote -= total
                base += qty

                sell_net = (qty * info["sell_price"]) - info["fee_sell"]
                pnl = sell_net - total

                pairs.append({
                    "type": "SELL->BUY",
                    "open_time": info["time_sell"],
                    "close_time": ts,
                    "open_price": info["sell_price"],
                    "close_price": px,
                    "open_grid": info["sell_grid"],
                    "close_grid": g,
                    "qty": qty,
                    "fee_open": info["fee_sell"],
                    "fee_close": fee,
                    "pnl": pnl
                })

                events.append({
                    "time": ts, "side": "BUY", "price": px, "grid": g,
                    "qty": qty, "fee_quote": fee,
                    "quote_after": quote, "base_after": base,
                    "kind": "SHORT_CLOSE", "pnl_pair": pnl
                })

                del short_open[sell_level]

        # APERTURAS (solo grids reales)
        for g in down_crossed:
            if g not in grids_real:
                continue
            if g >= start_price:
                continue

            gi = idx[g]
            if gi >= len(ladder) - 1:
                continue
            target = ladder[gi + 1]

            if g in long_open:
                continue

            px = exec_price(g, "BUY")
            qty = cfg.order_quote / px
            cost = qty * px
            fee = cost * cfg.fee_rate
            total = cost + fee

            if quote >= total:
                quote -= total
                base += qty

                long_open[g] = {
                    "buy_price": px,
                    "buy_grid": g,
                    "qty": qty,
                    "fee_buy": fee,
                    "time_buy": ts,
                    "target_level": target
                }

                events.append({
                    "time": ts, "side": "BUY", "price": px, "grid": g,
                    "qty": qty, "fee_quote": fee,
                    "quote_after": quote, "base_after": base,
                    "kind": "LONG_OPEN"
                })

        for g in up_crossed:
            if g not in grids_real:
                continue
            if g <= start_price:
                continue

            gi = idx[g]
            if gi <= 0:
                continue
            target = ladder[gi - 1]

            if g in short_open:
                continue

            px = exec_price(g, "SELL")
            qty = cfg.order_quote / px
            if base + 1e-12 < qty:
                continue

            proceeds = qty * px
            fee = proceeds * cfg.fee_rate
            net = proceeds - fee

            base -= qty
            quote += net

            short_open[g] = {
                "sell_price": px,
                "sell_grid": g,
                "qty": qty,
                "fee_sell": fee,
                "time_sell": ts,
                "target_level": target
            }

            events.append({
                "time": ts, "side": "SELL", "price": px, "grid": g,
                "qty": qty, "fee_quote": fee,
                "quote_after": quote, "base_after": base,
                "kind": "SHORT_OPEN"
            })

        prev_close = cl

    events_df = pd.DataFrame(events)
    pairs_df = pd.DataFrame(pairs)

    last_price = float(df["close"].iloc[-1])
    final_value = quote + base * last_price

    unrealized = 0.0
    for info in long_open.values():
        qty = info["qty"]
        buy_cost = qty * info["buy_price"] + info["fee_buy"]
        sell_gross_now = qty * last_price
        sell_fee_now = sell_gross_now * cfg.fee_rate
        sell_net_now = sell_gross_now - sell_fee_now
        unrealized += (sell_net_now - buy_cost)

    for info in short_open.values():
        qty = info["qty"]
        sell_net = (qty * info["sell_price"]) - info["fee_sell"]
        buy_gross_now = qty * last_price
        buy_fee_now = buy_gross_now * cfg.fee_rate
        buy_total_now = buy_gross_now + buy_fee_now
        unrealized += (sell_net - buy_total_now)

    realized = float(pairs_df["pnl"].sum()) if len(pairs_df) else 0.0
    total_pnl = realized + unrealized

    initial_value = quote0 + base0 * start_price

    summary = {
        "start_price": start_price,
        "midline": midline,
        "highest_buy": highest_buy,
        "lowest_sell": lowest_sell,
        "quote_end": quote,
        "base_end": base,
        "last_price": last_price,
        "final_value": final_value,
        "initial_value": initial_value,
        "realized_pnl_sum": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": total_pnl,
        "completed_pairs": int(len(pairs_df)),
        "total_fees_pairs": float(pairs_df[["fee_open", "fee_close"]].sum().sum()) if len(pairs_df) else 0.0,
        "open_longs": int(len(long_open)),
        "open_shorts": int(len(short_open)),
        "ladder": ladder,
        "grids_real": grids_real,
    }

    return summary, events_df, pairs_df, grids_real


# =========================================================
# PLOTLY PLOT (HTML)
# =========================================================
def build_plotly(df_utc, grids_real, events_df, pairs_df, cfg: GridConfig, midline=None, title="Grid Emulator (1m)"):
    tz_local = ZoneInfo(cfg.show_timezone)
    df = df_utc.copy()
    df.index = df.index.tz_convert(tz_local)

    fig = go.Figure()

    # precio (negro)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"],
        mode="lines", name="Price",
        line=dict(color="black", width=1.5)
    ))

    # líneas de grids (rojo)
    for g in sorted(grids_real):
        fig.add_hline(y=g, line=dict(color="red", width=1, dash="solid"), opacity=0.55)

    # start (azul)
    fig.add_hline(y=cfg.start_price, line=dict(color="blue", width=1.5), opacity=0.85)

    # midline (gris punteada)
    if midline is not None:
        fig.add_hline(y=midline, line=dict(color="gray", width=1.2, dash="dash"), opacity=0.9)

    # markers buy/sell
    if len(events_df):
        ev = events_df.copy()
        ev["time"] = pd.to_datetime(ev["time"], utc=True).dt.tz_convert(tz_local)
        buys = ev[ev["side"] == "BUY"]
        sells = ev[ev["side"] == "SELL"]

        fig.add_trace(go.Scatter(
            x=buys["time"], y=buys["price"],
            mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", size=10, color="red")
        ))
        fig.add_trace(go.Scatter(
            x=sells["time"], y=sells["price"],
            mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", size=10, color="green")
        ))

    # líneas punteadas por par
    if len(pairs_df):
        pr = pairs_df.copy()
        pr["open_time"] = pd.to_datetime(pr["open_time"], utc=True).dt.tz_convert(tz_local)
        pr["close_time"] = pd.to_datetime(pr["close_time"], utc=True).dt.tz_convert(tz_local)
        for _, r in pr.iterrows():
            fig.add_trace(go.Scatter(
                x=[r["open_time"], r["close_time"]],
                y=[r["open_price"], r["close_price"]],
                mode="lines",
                line=dict(color="gray", dash="dash", width=1),
                opacity=0.6,
                showlegend=False
            ))

    fig.update_layout(
        title=title,
        xaxis_title=f"Time ({cfg.show_timezone})",
        yaxis_title="Price",
        hovermode="x unified",
        height=600,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


# =========================================================
# HTML TEMPLATE
# =========================================================
TPL = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Grid Emulator</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body{font-family:Arial, sans-serif; margin:18px;}
    input, textarea{width:100%; padding:8px; margin:6px 0;}
    .row{display:grid; grid-template-columns:1fr 1fr; gap:12px;}
    .card{border:1px solid #ddd; border-radius:10px; padding:12px; margin-top:12px;}
    table{width:100%; border-collapse:collapse;}
    th,td{border-bottom:1px solid #eee; padding:6px; font-size:12px; text-align:left;}
    .btn{padding:10px 12px; background:#111; color:#fff; border:0; border-radius:8px; cursor:pointer;}
    .err{color:#b00020; white-space:pre-wrap;}
    .mono{font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; white-space:pre-wrap;}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
</head>
<body>
  <h2>Grid Emulator (web)</h2>
  <div class="card">
    <form action="/run" method="post">
      <div class="row">
        <div>
          <label>Start local (Lima) - YYYY-MM-DD HH:MM</label>
          <input name="start_local" value="{{start_local}}" />
        </div>
        <div>
          <label>End local (Lima) - YYYY-MM-DD HH:MM</label>
          <input name="end_local" value="{{end_local}}" />
        </div>
      </div>
      <div class="row">
        <div>
          <label>Symbol</label>
          <input name="symbol" value="{{symbol}}" />
        </div>
        <div>
          <label>Trading start price</label>
          <input name="start_price" value="{{start_price}}" />
        </div>
      </div>
      <div class="row">
        <div>
          <label>Capital USDT</label>
          <input name="capital" value="{{capital}}" />
        </div>
        <div>
          <label>Fee rate (0.001 = 0.1%)</label>
          <input name="fee_rate" value="{{fee_rate}}" />
        </div>
      </div>
      <label>Grids (separados por coma)</label>
      <textarea name="grids" rows="3">{{grids}}</textarea>
      <button class="btn" type="submit">Run</button>
    </form>
  </div>

  {% if error %}
    <div class="card err">{{error}}</div>
  {% endif %}

  {% if summary %}
    <div class="card">
      <h3>Resumen</h3>
      <div class="mono">{{summary_text}}</div>
    </div>
  {% endif %}

  {% if plot_div %}
    <div class="card">
      <h3>Plot</h3>
      {{plot_div | safe}}
    </div>
  {% endif %}

  {% if events_html %}
    <div class="card">
      <h3>Events (top)</h3>
      {{events_html | safe}}
    </div>
  {% endif %}

  {% if pairs_html %}
    <div class="card">
      <h3>Pares (top)</h3>
      {{pairs_html | safe}}
    </div>
  {% endif %}

</body>
</html>
"""

def parse_dt_local(s: str, tz_name="America/Lima") -> datetime:
    tz = ZoneInfo(tz_name)
    # espera "YYYY-MM-DD HH:MM"
    dt = datetime.strptime(s.strip(), "%Y-%m-%d %H:%M")
    return dt.replace(tzinfo=tz)

def parse_grids(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]

@app.get("/")
def index():
    # defaults (los tuyos)
    return render_template_string(
        TPL,
        start_local="2026-01-11 23:54",
        end_local=datetime.now(ZoneInfo("America/Lima")).strftime("%Y-%m-%d %H:%M"),
        symbol="BTCUSDT",
        start_price="92188.95",
        capital="106.95",
        fee_rate="0.001",
        grids="92908, 91508",
        error=None, summary=None, summary_text=None,
        plot_div=None, events_html=None, pairs_html=None
    )

@app.post("/run")
def run():
    try:
        tz_name = "America/Lima"
        start_local = parse_dt_local(request.form["start_local"], tz_name)
        end_local = parse_dt_local(request.form["end_local"], tz_name)
        start_utc = start_local.astimezone(ZoneInfo("UTC"))
        end_utc = end_local.astimezone(ZoneInfo("UTC"))

        symbol = request.form.get("symbol", "BTCUSDT").strip()
        start_price = float(request.form["start_price"])
        capital = float(request.form["capital"])
        fee_rate = float(request.form.get("fee_rate", "0.001"))
        grids = parse_grids(request.form["grids"])

        cfg = GridConfig(
            symbol=symbol,
            fee_rate=fee_rate,
            quote_start=capital,
            order_quote=capital / max(1, len(grids)),
            grids=grids,
            start_price=start_price,
            allow_multi_fills_per_bar=True,
            slippage=0.0,
            show_timezone=tz_name,
            init_mode="balanced",
        )

        df = fetch_klines_1m(cfg.symbol, start_utc, end_utc)
        summary, events_df, pairs_df, grids_real = simulate_grid_hilo(df, cfg)

        # texto resumen tipo consola
        summary_text = (
            f"Rango (Perú): {start_local.strftime('%Y-%m-%d %H:%M')} -> {end_local.strftime('%Y-%m-%d %H:%M')}\n"
            f"Start Price: {summary['start_price']:.2f}\n"
            f"Midline: {summary['midline'] if summary['midline'] is not None else 'N/A'}\n"
            f"Capital inicial: {summary['initial_value']:.2f}\n"
            f"Valor final: {summary['final_value']:.2f}\n"
            f"Pares completados: {summary['completed_pairs']}\n"
            f"PnL realizado: {summary['realized_pnl_sum']:+.4f}\n"
            f"Comisiones (pares): -{summary['total_fees_pairs']:.4f}\n"
            f"PnL no realizado: {summary['unrealized_pnl']:+.4f}\n"
            f"PnL total: {summary['total_pnl']:+.4f}\n"
            f"BTC inventario: {summary['base_end']:.6f}\n"
        )

        fig = build_plotly(
            df, grids_real, events_df, pairs_df, cfg,
            midline=summary["midline"],
            title=f"{cfg.symbol} Grid Emulator (1m) - Lima (HIGH/LOW)"
        )

        plot_div = fig.to_html(full_html=False, include_plotlyjs=False)

        # tablas (top N)
        tz_local = ZoneInfo(tz_name)
        ev = events_df.copy()
        if len(ev):
            ev["time_local"] = pd.to_datetime(ev["time"], utc=True).dt.tz_convert(tz_local)
        pr = pairs_df.copy()
        if len(pr):
            pr["open_time_local"] = pd.to_datetime(pr["open_time"], utc=True).dt.tz_convert(tz_local)
            pr["close_time_local"] = pd.to_datetime(pr["close_time"], utc=True).dt.tz_convert(tz_local)

        events_html = (ev.head(200).to_html(index=False) if len(ev) else "<div class='mono'>No hubo operaciones.</div>")
        pairs_html = (pr.head(200).to_html(index=False) if len(pr) else "<div class='mono'>No hubo pares completados.</div>")

        return render_template_string(
            TPL,
            start_local=request.form["start_local"],
            end_local=request.form["end_local"],
            symbol=symbol,
            start_price=request.form["start_price"],
            capital=request.form["capital"],
            fee_rate=request.form.get("fee_rate", "0.001"),
            grids=request.form["grids"],
            error=None,
            summary=summary,
            summary_text=summary_text,
            plot_div=plot_div,
            events_html=events_html,
            pairs_html=pairs_html,
        )

    except Exception as e:
        return render_template_string(
            TPL,
            start_local=request.form.get("start_local","2026-01-06 14:39"),
            end_local=request.form.get("end_local", datetime.now(ZoneInfo("America/Lima")).strftime("%Y-%m-%d %H:%M")),
            symbol=request.form.get("symbol","BTCUSDT"),
            start_price=request.form.get("start_price","92091.65"),
            capital=request.form.get("capital","107.06"),
            fee_rate=request.form.get("fee_rate","0.001"),
            grids=request.form.get("grids",""),
            error=str(e),
            summary=None, summary_text=None,
            plot_div=None, events_html=None, pairs_html=None
        )

if __name__ == "__main__":
    # Para que se vea desde tu celular en la misma red:
    # host="0.0.0.0" expone en tu LAN
    app.run(host="0.0.0.0", port=5000, debug=True)

