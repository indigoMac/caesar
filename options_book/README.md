# SPX put-spread delta hedge

Paper-only book that sells one SPX put credit spread and hedges the leftover delta with Micro E-mini S&P futures (MES). The ETF reconstitution study in `etf_rebalance/` is parked and is not used here.

This is not a claim that the rules make money. The bet is short put premium: you keep the credit if SPX stays above the short strike, and you can lose up to the spread width minus that credit if it does not. Flattening delta does not create the edge. Vega is added up on every cycle and is not traded.

A one-lot spread is on the order of 1–2 MES. Full-size ES is $50 per index point, so the same book is a fraction of one ES. The hedge symbol is `MES` until the book is large enough for ES. Dollar delta per index point is `quantity × model delta × 100`. Divide by 5 for MES, or by 50 for ES.

## Safety

Three checks have to pass before any order:

- `broker.submit_orders` is false in `config/paper.yaml`. The first runs connect, reconcile, and write the order they would send.
- The socket must be the paper Gateway at `127.0.0.1:4002`. A different host or port is refused unless `broker.live_enabled` is true and the environment variable `OPTIONS_BOOK_LIVE=1` is set. This version ships with `live_enabled: false`.
- A halt latch blocks new orders after a bad reconciliation, stale greeks on an open book, a daily loss at the configured limit, an exception, or a `data/HALT` file. Restarting does not clear the latch. `options-book clear-halt` clears it, and it refuses while the halt file is still there.

Startup compares broker positions with the audit log. An SPX option or MES/ES future that the log does not explain halts the book. Spreads are combo orders. A one-leg fill halts; the engine does not leg out of it.

A flat book with no usable quote does not latch. An open book with missing or stale greeks does.

## Run

Install IB Gateway for the paper account. Enable the API socket, set the port to 4002, and check "Download open orders on connection." Enable index-option and futures trading on the paper account. Live SPX greeks and MES quotes need the matching market-data lines. Delayed data is not used. Start Gateway yourself. This process does not log in and does not store a username or password.

```bash
cd options_book
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
options-book once --config config/paper.yaml
```

Read the log. When the intended combo looks right, set `broker.submit_orders: true` and run:

```bash
options-book run --config config/paper.yaml
```

The cycle is about once a minute, and only from 09:45 to 15:45 America/New_York on an NYSE session. Entry sells the 30–45 day SPXW put whose delta is closest to -0.16, and buys the put `width_points` lower, as one limit combo. The limit is the combo mid, negative on a buy, which is how IB encodes a credit. The order is cancelled if it is still working after `order_timeout_seconds`.

The spread is closed with the opposite combo at 50% of the credit, at 21 days to expiry, or when the loss reaches `loss_multiple_of_credit` times the credit. The future hedge is a limit at the inside quote, and only when net delta is outside a 1-contract band. `max_mes_position` is a hard cap. If the book needs more than that and would still sit outside the band, the engine halts instead of adding size.

The audit log is `data/state.sqlite`. To stop new orders immediately, create `data/HALT`. To trade again, remove that file and run `options-book clear-halt --config config/paper.yaml`. A halt exits with status 2.

Other positions in the account are left alone. Manual SPX, MES, or ES orders will latch the engine, because it cannot tell them from its own.
