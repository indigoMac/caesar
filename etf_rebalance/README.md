# Russell reconstitution event study

Research prototype for a point-in-time backtest of Russell index additions and deletions. It is not a trading system, and it does not estimate whether the strategy would have made money.

The checked-in market files under `data/raw/` are **synthetic fixtures**. They exercise the pipeline. They are not FTSE Russell announcements and they are not a price history. Replace them before reading any output as evidence.

## Research hypothesis

Public Russell reconstitution announcements may move prices between the announcement and the effective close, because index funds have to trade into the new weights.

- **H1.** Additions have positive abnormal returns from the first tradable session after the announcement through the rebalance close.
- **H2.** Deletions have negative abnormal returns over that same window.
- **H3.** Part of the pre-rebalance move reverses after the rebalance close.

The mechanism to test later is whether the abnormal return lines up with expected index demand divided by liquidity (`NetFlow / ADV`), inside each reconstitution, not across a pooled stock-level regression. This repository does not fit that flow model yet. It ranks names with an explicitly labelled proxy when `estimated_flow` is absent.

Nothing here is optimized. There is no machine-learning model, no parameter search, and no claim that a portfolio is profitable.

## Architecture

```
etf_rebalance/
  config/                 research.yaml, costs.yaml
  data/raw/               immutable inputs
  data/processed/         latest derived panel
  src/rebalance/
    calendar.py           XNYS sessions, next legal open
    data/                 security master, announcement CSV, raw prices
    events/               revision state and tradable signals
    features/             total return, liquidity, pressure
    research/             event panel, quintiles, bootstrap, runner
    portfolio/            weights, flat costs, buy-and-hold simulator
    reporting/            tables and matplotlib figures
  tests/
  notebooks/
```

Raw files are only read. Writers refuse a path under `data/raw/`.

A ticker is not an identifier. `security_id` comes from `data/raw/russell/sample_security_master.csv`, using the ticker interval that covers `published_at`. The sample master maps `OLD_A01` to `A01` through 2022 and `A01` after that.

The stored object is a public announcement, not the final membership list. Each row has:

`rebalance_id`, `security_id`, `ticker_published`, `company_name_published`, `index_name`, `action` (`ADD`, `DELETE`, or `NONE`), `published_at`, `effective_close`, `version`, `source_url`, `source_hash`, `retrieved_at`.

`event_state_asof(events, timestamp)` keeps rows with `published_at <= timestamp`, takes the latest version per security, and drops a security whose latest action is `NONE`. Earlier rows stay in the log. A June removal does not rewrite a May addition.

## Data requirements

The announcement loader does not know FTSE Russell's file layout. Normalize a real extract to this CSV before use:

`published_at`, `effective_close`, `ticker`, `company_name`, `index_name`, `action`, `version`, `rebalance_id`, `source_url`, `retrieved_at`

`published_at` and `retrieved_at` need a timezone offset. The loader will not assume one unless `naive_timestamps_tz` is set in config. `effective_close` is a calendar date and must be an exchange session. `retrieved_at` is required so the load is reproducible; it is not filled with `now()`.

Prices, raw and unadjusted:

`security_id`, `date`, `open_raw`, `high_raw`, `low_raw`, `close_raw`, `volume_raw`, `split_factor`, `dividend`

`split_factor` is 1 when the session is not an ex-split. `dividend` is cash per post-split share and 0 when there is none. Total return is computed beside these columns. Adjusted prices never replace them.

Optional fundamentals (`security_id`, `asof`, and any of `market_cap`, `delta_weight`, `estimated_flow`) and optional borrow (`security_id`, `asof`, `borrow_available`) are joined only on rows whose `asof` is at or before `signal_known_at`. A missing borrow file does not mean the name was easy to borrow.

TODO: map licensed FTSE Russell preliminary, updated, and final reconstitution files into the announcement schema. TODO: point `config/research.yaml` at a corporate-action-aware price history. Do not invent the missing history.

`data/raw/corporate_actions/sample_corporate_actions.csv` is the vendor-shaped split and dividend file for the fixture. The sample price file already contains those factors. `merge_corporate_actions` copies them onto bars without changing open, high, low, close, or volume, and it raises if the two sources disagree.

## Anti-look-ahead safeguards

Every tradable signal must satisfy:

```
feature_asof <= signal_known_at < earliest_trade_at
```

`earliest_trade_at` is the next regular NYSE open strictly after the announcement that determines the trade. Weekdays are not trading days. The calendar is `exchange-calendars` XNYS.

Consequences that the tests lock in:

- A Friday 17:00 America/New_York announcement cannot fill at Friday's close. On 24 May 2024 that open is Tuesday 28 May, because Monday is Memorial Day.
- An announcement at 11:00 cannot use that session's open. The same-morning open is already in the past.
- An announcement at 08:00, before the open, can trade that session's open.
- Good Friday and Independence Day are skipped.
- Features such as ADV use the last **completed** session at `signal_known_at`, then sessions t-25 through t-6. Bars after the signal do not change ADV.
- A fundamental dated after the signal is ignored, even if it is already in the file.
- A borrow flag dated after the signal is ignored.
- A name removed by a `NONE` row before the entry open is not traded. The original `ADD` remains in the raw file.

V0 never trades the same session's close. A later extension could allow a same-session close only with an explicit cutoff that is still after `signal_known_at`.

## Event-study definitions

The addition signal is `+1` and the deletion signal is `-1`. In the event panel the deletion sign is **not** applied. H2 is a statement about the stock's own abnormal return. The portfolio applies the sign by using a negative weight.

Entry is the first regular-session open after `signal_known_at`. Exit for the held strategy is the official close on `effective_close`.

On the entry session, `stock_return` is open-to-close (`close / open - 1`). The buyer at the open does not receive that session's dividend. Every other session is a close-to-close total return:

```
(close_t + dividend_t) / (close_{t-1} / split_factor_t) - 1
```

The benchmark uses the same basis on the same session.

```
AR = stock_return - benchmark_return
CAR[a, b] = sum of AR on the sessions from a through b
```

That CAR is a sum of simple returns. The portfolio return is a compounded buy-and-hold total return, then scaled by signed weights. Those two numbers are not the same object. A 10% per-name cap also means a five-name book is not fully invested, so the portfolio return is smaller than the average stock CAR.

`days_from_announcement` is the trading-session offset from the entry session, not from the calendar timestamp of the press release. `days_from_rebalance` is the offset from `effective_close`. Session offsets skip holidays.

Windows in `config/research.yaml`:

| Window | Sessions included |
| --- | --- |
| `ann_to_rebalance` | entry through rebalance close |
| `ann_plus_2_to_rebalance` | entry+2 through rebalance close |
| `ann_plus_5_to_rebalance` | entry+5 through rebalance close |
| `reb_minus_5_to_rebalance` | rebalance-5 through rebalance close |
| `rebalance_to_plus_1` … `plus_20` | the sessions **after** the close, from +1 through +N |

Post-rebalance windows start the session after the exit close, because H3 is about what happens after the strategy is out. A window that does not fit, or that has a missing bar, is null. Missing days are not filled and partial sums are not reported as complete.

Additions and deletions are summarized separately. The primary estimate equally weights reconstitutions. The bootstrap resamples whole `rebalance_id`s. The t interval uses `n_rebalances - 1` degrees of freedom. A clustered standard error of the pooled stock mean is stored as a secondary column and is not the headline estimate.

Quintiles are assigned inside each `(rebalance_id, action)` group. Q1 is lowest pressure and Q5 is highest. Fewer than five scored names leaves the group unranked.

Pressure, in order:

1. `estimated_flow / ADV20`, signed so additions want positive flow and deletions want negative flow. `pressure_kind = net_flow_over_adv20` and `pressure_is_proxy = false`.
2. Else `abs(delta_weight) / ADV20`, labelled `abs_delta_weight_over_adv20`.
3. Else `market_cap / ADV20`, labelled `market_cap_over_adv20`.

Proxy scores are never copied into `estimated_flow`. The intended flow, once the inputs exist, is `sum_j AUM_j * (new_weight_ij - old_weight_ij)`.

Costs are one-way and charged on the way in and the way out:

```
net = gross - entry_cost - exit_cost
```

Scenarios in `config/costs.yaml`: optimistic 5 bps, base 15, conservative 30, hostile 50. `FlatBpsCostModel.one_way_cost` is the extension point for spread, impact, and borrow fees.

Portfolio modes: long additions, short deletions, long/short, and long additions hedged with the benchmark ETF for the same gross. Sizing is equal weight or pressure weight, with a per-name cap, a gross-exposure cap, and an optional ADV participation cap that runs only when both `portfolio_nav` and `max_adv_participation` are set. Short books without a located borrow record are marked theoretical.

Chronological slices in the config are `development`, `validation`, and `out_of_sample`. The synthetic 2023 and 2024 dates fall in `out_of_sample`. The runner still describes them. It does not search parameters on any slice.

Each run writes `runs/<run_id>/run_manifest.json` with the run id, timestamp, git commit if this directory is inside a repository, the configuration, dataset version labels, file hashes, date range, entry rule, exit rule, and cost assumptions. `data/processed/event_panel.parquet` is the latest derived panel.

## How to run tests

```bash
cd etf_rebalance
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The tests cover the Friday-close and holiday calendar, the May 22 / May 29 / June 5 revision sequence, return and split math, feature and fill timestamps, transaction costs, event-relative session counts, and a bootstrap that resamples whole rebalance ids. `tests/test_sample_study.py` runs the fixture study and checks that raw files are unchanged.

## How to run a sample event study

```bash
python scripts/build_sample_fixtures.py   # rewrites the synthetic CSVs
python -m rebalance.research.run
```

Notebooks, in order:

1. `notebooks/01_event_data_audit.ipynb` — revision log versus `event_state_asof`
2. `notebooks/02_event_study.ipynb` — panel and window CARs
3. `notebooks/03_pressure_signal.ipynb` — proxy pressure and quintiles
4. `notebooks/04_portfolio_backtest.ipynb` — costs and the theoretical short flag

Regenerate the notebooks' inputs by running the study first, or run the notebooks themselves; they load the fixtures directly. Treat every printed number as a pipeline check.

## Current limitations

- No real Russell membership history is included. Sample output is not an estimate of H1, H2, or H3.
- Two reconstitutions cannot support an interval. Read `n_rebalances` before reading a confidence interval. `ci_excludes_zero` on the fixture is not a finding.
- With five names and a 10% cap, pressure weights hit the cap and match equal weights. The unit test covers unsaturated pressure weights.
- V0 holds a name through the rebalance close even if a later revision removes it after the entry. Removals published before the entry open are excluded. Early exit on a post-entry revision is not implemented.
- Same-session close execution is not implemented.
- The deletion panel does not model borrow fees, recalls, or locate availability. Short results are theoretical unless a borrow file is supplied.
- Costs are flat basis points, not spread or impact.
- `market_cap / ADV` is a size or liquidity proxy, not predicted index demand.
- CAR and portfolio return answer different accounting questions. Do not reconcile them by forcing one number.
- There is no optimizer. Do not add one that searches the out-of-sample slice.

## Next development steps

1. Ingest real point-in-time Russell preliminary and update files, including a source hash of the publisher file.
2. Attach a licensed price and corporate-action history, still storing raw bars.
3. Add point-in-time index weights and fund AUM, and fill `estimated_flow` without touching the proxy columns.
4. Exit at the next open when a revision cancels a name after entry.
5. Add borrow, spread, and impact models behind `CostModel`.
6. Only after those inputs exist, read the rebalance-clustered estimates. Do not tune windows or weights on `out_of_sample`.
