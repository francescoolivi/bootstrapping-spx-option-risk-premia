# Data

The input data are **not included** in this repository. The OptionMetrics IvyDB files
are licensed through Wharton Research Data Services (WRDS) and may not be
redistributed. Users with a WRDS subscription can rebuild them as described below.
Place the five CSV files in this folder (`data/`) with exactly these names. Column names
must also match.

Sample used in the thesis: **4 January 1996 – 29 August 2025** (7,463 trading days),
S&P 500 index, OptionMetrics `secid = 108105`.

| File | Source | Columns used |
|---|---|---|
| `optionmetrics_volatilitysurface.csv` | WRDS → OptionMetrics IvyDB US → *Volatility Surface* (`optionm.vsurfd*`), `secid = 108105` | `date, days, delta, impl_volatility, cp_flag` (the full extract also contains `secid, impl_strike, impl_premium, dispersion, ticker, index_flag`) |
| `optionmetrics_spxprices.csv` | WRDS → OptionMetrics → *Security Prices* (`optionm.secprd*`), `secid = 108105` | `date, close, return` (`return` is the simple daily return) |
| `optionmetrics_yieldcurve.csv` | WRDS → OptionMetrics → *Zero Coupon Yield Curve* (`optionm.zerocd`) | `date, days, rate` (`rate` in annualized %) |
| `optionmetrics_idxdvd.csv` | WRDS → OptionMetrics → *Index Dividend Yield* (`optionm.idxdvd`), `secid = 108105` | `secid, date, expiration, rate` (loaded for reference only; not used in pricing) |
| `cboe_VIX.csv` | CBOE VIX historical data (cboe.com, *VIX Historical Data*), or WRDS → CBOE Indexes | `Date, vix` (daily VIX close, in index points) |

Notes:

- **Volatility surface.** The code uses 5 maturities (30, 91, 182, 273, 365 days) and
  5 deltas: put deltas −10 and −25, call deltas +50, +25 and +10. OptionMetrics stores put
  deltas as negative numbers (e.g. `-10`) and call deltas as positive numbers. The
  extract must contain at least these 25 (delta, days, cp_flag) nodes.
- **VIX.** If you download `VIX_History.csv` from CBOE, keep the `DATE` and `CLOSE`
  columns and rename them to `Date` and `vix`. Duplicate dates are dropped
  automatically.
- A shorter or more recent sample also works. The rolling exercise always starts at
  the sample midpoint, so the number of test dates and the results will differ from
  those reported in the thesis.
