"""
run_all.py — headless reproduction of every table and figure.

Bootstrapping Conditional Option Risk Premia on S&P 500 Index Options
(Francesco Olivi, Master's thesis, Bocconi University), implementing
Wu (2025), "Bootstrapping Option Risk Premiums".

Runs the same pipeline as notebooks/bootstrapping_option_risk_premia.ipynb
(same functions, same seeds) and additionally saves every figure plus the
robustness diagnostics of Section 15.

Input : ../data/   (see data/README.md — OptionMetrics data is licensed via WRDS)
Output: ../output/ (tables tab_*.csv, figures fig_*.png, rolling_covariance_matrices.npz)

Usage : python3 scripts/run_all.py        (about 6 minutes; ~95% is the Section 11 rolling loop)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # headless: no windows are opened
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import norm
from scipy.stats import norm as sp_norm
from scipy.interpolate import interp1d, griddata
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, 'data')
OUT    = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)

def save_fig(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → saved: {name}")

def save_csv(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path)
    print(f"  → saved: {name}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 1. DATA LOADING ===")

vol_surface = pd.read_csv(
    os.path.join(DATA, 'optionmetrics_volatilitysurface.csv'),
    parse_dates=['date'], low_memory=False
)
spx_prices = pd.read_csv(
    os.path.join(DATA, 'optionmetrics_spxprices.csv'),
    parse_dates=['date'], low_memory=False
)

print(f"Vol surface shape: {vol_surface.shape}")
print(f"SPX prices shape:  {spx_prices.shape}")

# ── Cleaning ─────────────────────────────────────────────────────────────────
vol_clean = vol_surface[vol_surface['impl_volatility'].notna()].copy()

# ── ATM IV 30d, merged with SPX ──────────────────────────────────────────────
atm_iv = vol_clean[
    (vol_clean['delta'] == 50) &
    (vol_clean['days']  == 30) &
    (vol_clean['cp_flag'] == 'C')
][['date', 'impl_volatility']].set_index('date').sort_index()
atm_iv.columns = ['IV_ATM_30d']

spx = spx_prices[['date', 'close', 'return']].set_index('date').sort_index()
df  = spx.join(atm_iv, how='inner')

# ── Figure: SPX and ATM IV 30d ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(df.index, df['close'], linewidth=0.7, color='black')
axes[0].set_ylabel('S&P 500 Level')
axes[0].set_title('S&P 500 Price and ATM Implied Volatility (30-day)')
axes[1].plot(df.index, df['IV_ATM_30d'] * 100, linewidth=0.7, color='darkred')
axes[1].set_ylabel('IV ATM 30d (%)')
axes[1].set_xlabel('Date')
plt.tight_layout()
save_fig('fig_spx_atm_iv.png')

# ══════════════════════════════════════════════════════════════════════════════
# 2. RETURN INNOVATION  ε  (Wu 2025, eq. 21)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. RETURN INNOVATION ε ===")

dt = 1 / 252
# OptionMetrics' 'return' column is a SIMPLE return (it equals close.pct_change()),
# so np.log(1 + df['return']) would also be correct. The log-return is computed
# directly from prices (numerically identical, max diff ~4e-5) so as not to depend
# on that column.
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['epsilon']    = df['log_return'] / (df['IV_ATM_30d'].shift(1) * np.sqrt(dt))
df = df.dropna(subset=['epsilon'])

print(f"Obs: {len(df)}")
print(f"Mean={df['epsilon'].mean():.4f}  Std={df['epsilon'].std():.4f}  "
      f"Skew={df['epsilon'].skew():.4f}  Kurt={df['epsilon'].kurtosis():.4f}")

# ── Figure: ε diagnostics ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].plot(df.index, df['epsilon'], linewidth=0.3, alpha=0.7)
axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Return Innovation ε')
x_range = np.linspace(-8, 8, 300)
axes[0, 1].hist(df['epsilon'], bins=200, density=True, alpha=0.7, label='Empirical ε')
axes[0, 1].plot(x_range, norm.pdf(x_range, 0, df['epsilon'].std()), 'r-', lw=2, label='Normal')
axes[0, 1].set_xlim(-8, 8)
axes[0, 1].legend()
axes[0, 1].set_title('Distribution of ε')
plot_acf(df['epsilon'],    lags=40, ax=axes[1, 0], title='ACF of ε')
plot_acf(df['epsilon']**2, lags=40, ax=axes[1, 1], title='ACF of ε²')
plt.tight_layout()
save_fig('fig_return_innovation_epsilon.png')

# ══════════════════════════════════════════════════════════════════════════════
# 3. IV INNOVATION  ζ  —  DRIFT FUNCTION  (Wu 2025, eq. 2–3, sec. 3.4)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. IV INNOVATION ζ — DRIFT FUNCTION ===")

df['ln_IV']  = np.log(df['IV_ATM_30d'])
theta        = df['ln_IV'].mean()
sigma        = df['ln_IV'].std()
df['x']      = (theta - df['ln_IV']) / sigma
df['dln_IV'] = df['ln_IV'].diff()
df_iv        = df.dropna(subset=['dln_IV', 'x']).copy()

def estimate_drift_function(x, y, bandwidth=7, n_grid=101):
    """Local linear regression with a monotonicity constraint (Wu 2025, sec. 3.4)."""
    x_grid = np.linspace(x.min(), x.max(), n_grid)
    mu_raw = np.zeros(n_grid)
    for i, x0 in enumerate(x_grid):
        u = (x - x0) / bandwidth
        w = np.exp(-0.5 * u**2)
        X_local = np.column_stack([np.ones(len(x)), x - x0])
        Xw = X_local * w[:, None]  # vectorized — avoids n×n np.diag matrix
        try:
            beta = np.linalg.solve(X_local.T @ Xw, Xw.T @ y)
            mu_raw[i] = beta[0]
        except np.linalg.LinAlgError:
            mu_raw[i] = 0.0
    mu_mono = mu_raw.copy()
    ci = np.argmin(np.abs(x_grid))
    for i in range(ci + 1, n_grid):
        mu_mono[i] = max(mu_mono[i], mu_mono[i - 1])
    for i in range(ci - 1, -1, -1):
        mu_mono[i] = min(mu_mono[i], mu_mono[i + 1])
    return x_grid, mu_raw, mu_mono

x_vals = df_iv['x'].shift(1).dropna().values
y_vals = df_iv['dln_IV'].iloc[1:].values
x_grid, mu_raw, mu_mono = estimate_drift_function(x_vals, y_vals)

mu_interp         = interp1d(x_grid, mu_mono, kind='linear', fill_value='extrapolate')
df_iv['mu_x']     = mu_interp(df_iv['x'].shift(1))
df_iv['zeta']     = df_iv['dln_IV'] - df_iv['mu_x']
df_iv             = df_iv.dropna(subset=['zeta'])

# ── Figure: drift function and ζ ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].scatter(x_vals, y_vals * 100, s=3, alpha=0.12, color='gray',
                   zorder=0, label=r'Observed $(x_t,\ v_{t+1}-v_t)$')
axes[0, 0].plot(x_grid, mu_raw * 100, 'b--', alpha=0.5, label='Raw')
axes[0, 0].plot(x_grid, mu_mono * 100, 'r-',  linewidth=2, label='Monotonicity-constrained')
axes[0, 0].set_ylim(-6, 6)
axes[0, 0].axhline(0, color='gray', linestyle=':', alpha=0.5)
axes[0, 0].axvline(0, color='gray', linestyle=':', alpha=0.5)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('μ(x), %')
axes[0, 0].set_title('Drift Function μ(x) — IV ATM 30d')
axes[0, 0].legend()
axes[0, 1].plot(df_iv.index, df_iv['zeta'], linewidth=0.3, alpha=0.7)
axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0, 1].set_title('IV Innovation ζ (ATM 30d)')
plot_acf(df_iv['zeta'],         lags=40, ax=axes[1, 0], title='ACF of ζ')
plot_acf(df_iv['dln_IV'].dropna(), lags=40, ax=axes[1, 1], title='ACF of raw Δln(IV)')
plt.tight_layout()
save_fig('fig_drift_function_atm_30d.png')

# ══════════════════════════════════════════════════════════════════════════════
# 4. SURFACE INNOVATIONS  —  25-contract grid (Wu delta convention)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 4. SURFACE INNOVATIONS (25 series) ===")

SELECTED_MATURITIES = [30, 91, 182, 273, 365]

# Mapping: (wu_delta, om_delta, cp_flag)
SELECTED_DELTAS = [
    (10,  -10, 'P'),   # deep OTM put
    (25,  -25, 'P'),   # OTM put
    (50,   50, 'C'),   # ATM call
    (75,   25, 'C'),   # OTM call  ≡  Wu delta 75
    (90,   10, 'C'),   # deep OTM call  ≡  Wu delta 90
]

def make_series_id(wu_delta, days):
    return f"d{wu_delta}_{days}d"

grid_list = []
for days in SELECTED_MATURITIES:
    for wu_delta, om_delta, cp_flag in SELECTED_DELTAS:
        grid_list.append({
            'days':      days,
            'wu_delta':  wu_delta,
            'om_delta':  om_delta,
            'cp_flag':   cp_flag,
            'series_id': make_series_id(wu_delta, days),
        })
grid_df = pd.DataFrame(grid_list)
print(f"Grid: {len(grid_df)} series ({len(SELECTED_MATURITIES)} maturities × {len(SELECTED_DELTAS)} deltas)")

# ── IV panel ───────────────────────────────────────────────────────────────────
iv_panel = pd.DataFrame()
for _, row in grid_df.iterrows():
    mask = (
        (vol_clean['days']     == row['days'])     &
        (vol_clean['delta']    == row['om_delta'])  &
        (vol_clean['cp_flag']  == row['cp_flag'])
    )
    temp = vol_clean[mask][['date', 'impl_volatility']].copy()
    temp = temp.rename(columns={'impl_volatility': row['series_id']}).set_index('date')
    iv_panel = temp if iv_panel.empty else iv_panel.join(temp, how='outer')
iv_panel = iv_panel.sort_index()

print(f"IV panel shape: {iv_panel.shape}")
print(f"Missing per series:\n{iv_panel.isna().sum().to_string()}")

# ── Drift and innovations for the 25 series (full sample) ────────────────────────────
drift_functions = {}
iv_stats        = {}
zeta_panel      = pd.DataFrame(index=iv_panel.index)

print(f"\n{'Series':<18} {'theta':>8} {'sigma':>8} {'mu(0)':>10} {'zeta_mean':>10} {'zeta_std':>10}")
print("-" * 70)

for series_id in grid_df['series_id']:
    iv_series = iv_panel[series_id].dropna()
    if len(iv_series) < 500:
        print(f"{series_id:<18} SKIPPED (obs={len(iv_series)})")
        continue

    ln_iv       = np.log(iv_series)
    theta_s     = ln_iv.mean()
    sigma_s     = ln_iv.std()
    x_s         = (theta_s - ln_iv) / sigma_s
    dln_iv      = ln_iv.diff()

    x_shifted   = x_s.shift(1).dropna()
    common_idx  = x_shifted.index.intersection(dln_iv.dropna().index)
    x_g, _, mu_m = estimate_drift_function(
        x_shifted.loc[common_idx].values,
        dln_iv.loc[common_idx].values
    )

    mu_interp_s      = interp1d(x_g, mu_m, kind='linear', fill_value='extrapolate')
    x_all_shifted    = x_s.shift(1).dropna()
    dln_aligned      = dln_iv.loc[x_all_shifted.index].dropna()
    common           = x_all_shifted.index.intersection(dln_aligned.index)
    zeta_s           = dln_aligned.loc[common] - pd.Series(
                           mu_interp_s(x_all_shifted.loc[common].values), index=common)

    drift_functions[series_id] = (x_g, mu_m)
    iv_stats[series_id]        = (theta_s, sigma_s)
    zeta_panel[series_id]      = zeta_s

    ci = np.argmin(np.abs(x_g))
    print(f"{series_id:<18} {theta_s:>8.4f} {sigma_s:>8.4f} "
          f"{mu_m[ci]:>10.6f} {zeta_s.mean():>10.6f} {zeta_s.std():>10.6f}")

# ── Joint innovations ──────────────────────────────────────────────────────
innovations       = df[['epsilon']].copy().join(zeta_panel, how='inner')
innovations_clean = innovations.dropna()
print(f"\nJoint innovations shape: {innovations_clean.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 4b. VIX, REALIZED VARIANCE and the CONVENTIONAL VRP (BTZ 2009 benchmark)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 4b. VIX + RV + CONVENTIONAL VRP ===")

# Monthly realized variance (rolling 22 trading days)
df['rv_sq']    = df['log_return'] ** 2
df['RV_22d']   = df['rv_sq'].rolling(window=22, min_periods=18).sum()
df['RV_annual'] = df['RV_22d'] * (252 / 22)

# VIX from cboe_VIX.csv. The file contains 2 duplicated dates: they are dropped before
# the join, otherwise the left-join would silently duplicate those rows of df (and any
# later .reindex() on df's index would fail).
vix_raw = pd.read_csv(
    os.path.join(DATA, 'cboe_VIX.csv'),
    parse_dates=['Date']
).rename(columns={'Date': 'date', 'vix': 'VIX'}).sort_values('date').drop_duplicates('date').set_index('date').sort_index()
df = df.join(vix_raw[['VIX']], how='left')
df['VIX2'] = (df['VIX'] / 100) ** 2   # annualized variance

# Conventional VRP, with both sign conventions made explicit. VRP_seller (= VIX2 - RV)
# is the reference convention (positive on average: sellers of protection earn a premium).
df['VRP_seller']       = df['VIX2'] - df['RV_annual']
df['VRP_buyer']        = df['RV_annual'] - df['VIX2']
df['VRP_conventional'] = df['VRP_seller']

vrp_clean = df[['VIX','RV_annual','VRP_conventional']].dropna()
print(f"VIX obs: {df['VIX'].notna().sum()}  |  RV obs: {df['RV_annual'].notna().sum()}")
print(f"VRP obs: {len(vrp_clean)}")
print(f"VRP mean: {vrp_clean['VRP_conventional'].mean():.6f}  "
      f"({vrp_clean['VRP_conventional'].mean()*10000:.1f} bps)")
print(f"VRP % negative: {(vrp_clean['VRP_conventional'] < 0).mean()*100:.1f}%")

# Summary table — use describe() on each column separately to avoid index alignment issues
vrp_df = df[['VIX','RV_annual','VRP_conventional']].dropna().copy()
vrp_df['RVol_22d_pct']  = np.sqrt(vrp_df['RV_annual'].clip(lower=0)) * 100
vrp_df['VRP_bps']       = vrp_df['VRP_conventional'] * 10000
vrp_summary = vrp_df[['VIX','RVol_22d_pct','VRP_bps']].rename(
    columns={'VIX':'VIX (%)','RVol_22d_pct':'RVol 22d (%)','VRP_bps':'VRP (bps var)'}
).describe().round(4)
save_csv(vrp_summary, 'tab_vrp_summary.csv')

# Figure: VIX vs. realized volatility + VRP
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(df.index, df['VIX'],
             linewidth=0.7, color='darkblue', label='VIX (%)')
axes[0].plot(df.index, np.sqrt(df['RV_annual'].clip(lower=0)) * 100,
             linewidth=0.7, color='darkred', alpha=0.7, label='RVol 22d (%)')
axes[0].set_ylabel('%')
axes[0].legend()
axes[0].set_title('VIX and Annualized Realized Volatility')
axes[1].fill_between(df.index, df['VRP_conventional'].fillna(0) * 10000,
                     0, alpha=0.4, color='darkgreen')
axes[1].plot(df.index, df['VRP_conventional'] * 10000,
             linewidth=0.5, color='darkgreen')
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_ylabel('VRP (bps of variance)')
axes[1].set_xlabel('Date')
axes[1].set_title('Conventional VRP = VIX² − RV (BTZ 2009 benchmark)')
plt.tight_layout()
save_fig('fig_vrp_conventional.png')

# ══════════════════════════════════════════════════════════════════════════════
# 5. ACF(ζ²) DIAGNOSTICS + POWER COEFFICIENT β
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 5. ACF(ζ²) DIAGNOSTICS + POWER COEFFICIENT β ===")

# Figure: ACF of squared zeta
check_series = ['d10_30d', 'd50_30d', 'd90_30d',
                'd10_365d', 'd50_365d', 'd90_365d']
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('ACF of Squared IV Innovations (ζ²) — Selected Series', fontsize=14)
for ax, sid in zip(axes.flat, check_series):
    if sid in innovations_clean.columns:
        acf_vals = acf(innovations_clean[sid]**2, nlags=20)
        ax.bar(range(21), acf_vals, color='steelblue', alpha=0.7)
        ax.set_title(sid)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_ylim(-0.15, 0.4)
plt.tight_layout()
save_fig('fig_acf_zeta_squared.png')

# Table: power coefficient β
print(f"\n{'Series':<18} {'beta':>8}")
print("-" * 28)
beta_records = []
for _, row in grid_df.iterrows():
    sid    = row['series_id']
    iv_s   = iv_panel[sid].dropna()
    if len(iv_s) < 100:
        continue
    # Wu (2025): ΔV must be level change, not log-change
    dV = iv_s.diff().dropna()
    monthly_var   = dV.groupby(pd.Grouper(freq='ME')).apply(
                        lambda x: x.var() if len(x) >= 15 else np.nan).dropna()
    monthly_level = iv_s.groupby(pd.Grouper(freq='ME')).mean().loc[monthly_var.index]
    if len(monthly_var) < 20:
        continue
    y = np.log(monthly_var.values)
    x = np.log(monthly_level.values)
    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() < 20:
        continue
    slope = np.polyfit(x[valid], y[valid], 1)[0]
    beta  = slope / 2
    beta_records.append({'series_id': sid, 'wu_delta': row['wu_delta'],
                         'days': row['days'], 'beta': beta})
    print(f"{sid:<18} {beta:>8.3f}")

beta_df = pd.DataFrame(beta_records)
if not beta_df.empty:
    print(f"\nMean β: {beta_df['beta'].mean():.3f}  "
          f"[{beta_df['beta'].min():.3f}, {beta_df['beta'].max():.3f}]")
    save_csv(beta_df.set_index('series_id'), 'tab_beta_estimates.csv')

# ══════════════════════════════════════════════════════════════════════════════
# 6. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 6. DESCRIPTIVE STATISTICS ===")

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests

def lb_q20_sq(s):
    """Ljung-Box Q(20) and p-value on the squared series -- a volatility-
    clustering / residual-ARCH diagnostic (relevant for log-return and
    epsilon; for the IV level it is reported for formatting consistency but
    is not the diagnostic of interest)."""
    row = acorr_ljungbox(s.dropna() ** 2, lags=[20], return_df=True).iloc[0]
    return row['lb_stat'], row['lb_pvalue']

def ac1_sq(s):
    """AC(1) of the squared series (complements the AC(1) of the level)."""
    return (s.dropna() ** 2).autocorr(1)

stats_vars = {
    'S&P 500 Log Return (daily)': df['log_return'],
    'IV ATM 30d (level)':         df['IV_ATM_30d'],
    'Return Innovation epsilon':  df['epsilon'],
}
stats_rows = {}
for name, s in stats_vars.items():
    lb_stat, lb_pvalue = lb_q20_sq(s)
    stats_rows[name] = {
        'N obs':      int(s.notna().sum()),
        'Mean':       s.mean(),
        'Median':     s.median(),
        'Std':        s.std(),
        'Min':        s.min(),
        'Max':        s.max(),
        'Skewness':   s.skew(),
        'Kurtosis':   s.kurtosis(),
        'AC(1)':      s.autocorr(1),
        'AC(5)':      s.autocorr(5),
        'AC1_sq':     ac1_sq(s),
        'LB_Q20_sq':  lb_stat,
        'LB_Q20_sq_pvalue': lb_pvalue,
    }
stats_table = pd.DataFrame(stats_rows).T.round(4)
print(stats_table.to_string())
save_csv(stats_table, 'tab_summary_statistics.csv')

# IV mean / std / AC(1) by delta × maturity
pivot_data, pivot_std_data, pivot_ac_data = [], [], []
for _, row in grid_df.iterrows():
    s = iv_panel[row['series_id']].dropna()
    pivot_data.append(    {'wu_delta': row['wu_delta'], 'days': row['days'], 'v': s.mean() * 100})
    pivot_std_data.append({'wu_delta': row['wu_delta'], 'days': row['days'], 'v': s.std()  * 100})
    pivot_ac_data.append( {'wu_delta': row['wu_delta'], 'days': row['days'], 'v': s.autocorr(1)})

def make_pivot(records, val_name='Put Delta', col_name='Maturity (days)'):
    df_ = pd.DataFrame(records)
    p   = df_.pivot(index='wu_delta', columns='days', values='v')
    p.index.name   = val_name
    p.columns.name = col_name
    return p

iv_mean_tbl = make_pivot(pivot_data).round(2)
iv_std_tbl  = make_pivot(pivot_std_data).round(2)
iv_ac1_tbl  = make_pivot(pivot_ac_data).round(4)

print("\n--- IV Mean (%) ---")
print(iv_mean_tbl.to_string())
print("\n--- IV Std (%) ---")
print(iv_std_tbl.to_string())
print("\n--- IV AC(1) ---")
print(iv_ac1_tbl.to_string())

save_csv(iv_mean_tbl, 'tab_iv_mean_surface.csv')
save_csv(iv_std_tbl,  'tab_iv_std_surface.csv')
save_csv(iv_ac1_tbl,  'tab_iv_ac1_surface.csv')

# ══════════════════════════════════════════════════════════════════════════════
# 7. SURFACE FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 7. SURFACE FIGURES ===")

# Figure: IV time series, all 25 series
fig, ax = plt.subplots(figsize=(14, 6))
for sid in grid_df['series_id']:
    ax.plot(iv_panel.index, iv_panel[sid] * 100, linewidth=0.3, alpha=0.4, color='gray')
ax.plot(iv_panel.index, iv_panel['d50_30d']  * 100, linewidth=1.2, color='black',  label='ATM 1m')
ax.plot(iv_panel.index, iv_panel['d50_365d'] * 100, linewidth=1.2, color='red', linestyle='--', label='ATM 12m')
ax.set_ylabel('Implied Volatility (%)')
ax.set_xlabel('Date')
ax.set_title('S&P 500 Option Implied Volatility Surface — All 25 Series')
ax.legend()
plt.tight_layout()
save_fig('fig_iv_timeseries.png')

# Figure: ATM drift function at different maturities
fig, ax = plt.subplots(figsize=(10, 6))
colors_df = ['black', 'blue', 'green', 'red', 'purple']
styles_df = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
for i, (days, label) in enumerate(zip(SELECTED_MATURITIES, ['1m', '3m', '6m', '9m', '12m'])):
    sid = make_series_id(50, days)
    if sid in drift_functions:
        x_g, mu_m = drift_functions[sid]
        ax.plot(x_g, mu_m * 100, color=colors_df[i], linestyle=styles_df[i],
                linewidth=2, label=label)
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('x (standardized deviation)')
ax.set_ylabel('μ(x), %')
ax.set_title('Implied Volatility Drift Function — ATM at Different Maturities')
ax.legend()
plt.tight_layout()
save_fig('fig_drift_function_maturities.png')

# Figure: ACF of return² vs. epsilon²
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
acf_ret2 = acf(df['log_return'].dropna()**2, nlags=20)
acf_eps2 = acf(innovations_clean['epsilon']**2,  nlags=20)
axes[0].bar(range(21), acf_ret2, color='steelblue', alpha=0.7)
axes[0].set_title('ACF of Squared Log Returns')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Autocorrelation')
axes[0].axhline(0, color='gray', linestyle='--')
axes[1].bar(range(21), acf_eps2, color='darkred', alpha=0.7)
axes[1].set_title('ACF of Squared Return Innovations')
axes[1].set_xlabel('Lag')
axes[1].axhline(0, color='gray', linestyle='--')
plt.suptitle('Stochastic Volatility Removal: Raw Returns vs Scaled Innovations', fontsize=13)
plt.tight_layout()
save_fig('fig_acf_epsilon_vs_return.png')
print(f"  AC(1) return²={acf_ret2[1]:.4f}  epsilon²={acf_eps2[1]:.4f}  "
      f"reduction {(1 - acf_eps2[1]/acf_ret2[1])*100:.1f}%")

# Figure: mean IV heatmap
fig, ax = plt.subplots(figsize=(8, 5))
heatmap_data = iv_mean_tbl.values
im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(SELECTED_MATURITIES)))
ax.set_xticklabels([f'{d}d' for d in SELECTED_MATURITIES])
ax.set_yticks(range(5))
ax.set_yticklabels([10, 25, 50, 75, 90])
ax.set_xlabel('Maturity')
ax.set_ylabel('Put Delta (Wu convention)')
ax.set_title('Average Implied Volatility Surface (%)')
for i in range(5):
    for j in range(len(SELECTED_MATURITIES)):
        v = heatmap_data[i, j]
        ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9,
                color='white' if v > heatmap_data.mean() else 'black')
plt.colorbar(im, ax=ax, label='IV (%)')
plt.tight_layout()
save_fig('fig_iv_heatmap.png')

# ══════════════════════════════════════════════════════════════════════════════
# 7b. RISK-FREE RATE
# ══════════════════════════════════════════════════════════════════════════════
# Everything is expressed in forward units (F0 = 100, Wu 2025): the dividend yield
# is already absorbed in the forward (standard Black-76 result) and does not enter
# the pricing separately. The rate r is needed only to discount the option value
# from forward value to present value.
print("\n=== 7b. RISK-FREE RATE (OptionMetrics yield curve) ===")
yield_curve = pd.read_csv(os.path.join(DATA, 'optionmetrics_yieldcurve.csv'), parse_dates=['date'])

def get_risk_free_rate(date, tau_years, yield_curve):
    """Interpolate r(tau) (decimal) from the OptionMetrics yield curve at the closest available date (<= date)."""
    days_target = tau_years * 365.0
    avail_dates = yield_curve.loc[yield_curve['date'] <= date, 'date']
    use_date    = avail_dates.max() if not avail_dates.empty else yield_curve['date'].min()
    curve       = yield_curve[yield_curve['date'] == use_date].sort_values('days')
    r_pct       = np.interp(days_target, curve['days'].values, curve['rate'].values)
    return r_pct / 100.0

print(f"Yield curve: {yield_curve['date'].min().date()} - {yield_curve['date'].max().date()}, "
      f"{yield_curve['days'].nunique()} tenors available")

# ══════════════════════════════════════════════════════════════════════════════
# 8. BOOTSTRAP PATHS  (Wu 2025, eq. 4–7)  —  single illustrative test date
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 8. BOOTSTRAP PATHS ===")

N_PATHS      = 10000
HORIZON_DAYS = 5
F0           = 100.0
ATM_SERIES   = 'd50_30d'

# Test date: first Wednesday on or after the sample midpoint
all_dates        = innovations_clean.index
wednesdays       = all_dates[all_dates.weekday == 2]
midpoint         = all_dates[len(all_dates) // 2]
test_date        = wednesdays[wednesdays >= midpoint][0]
print(f"Test date: {test_date.strftime('%Y-%m-%d')}")

# IV at the test date
iv_at_test = {}
for sid in grid_df['series_id']:
    if test_date in iv_panel.index and pd.notna(iv_panel.loc[test_date, sid]):
        iv_at_test[sid] = iv_panel.loc[test_date, sid]
    else:
        iv_at_test[sid] = iv_panel[sid].loc[:test_date].dropna().iloc[-1]

print(f"ATM IV 30d at test date: {iv_at_test[ATM_SERIES]*100:.1f}%")

# No look-ahead: the zeta panel used for RESAMPLING is rebuilt from observations up
# to test_date only; otherwise theta/sigma/drift would "see" future data even for the
# pre-test_date historical innovations. innovations_clean (full sample) is used ONLY
# for descriptive statistics, never for the out-of-sample bootstrap.
iv_stats_local        = {}
drift_functions_local = {}
zeta_panel_local       = pd.DataFrame(index=iv_panel.index)
for sid in grid_df['series_id']:
    iv_hist     = iv_panel[sid].loc[:test_date].dropna()
    ln_iv_hist  = np.log(iv_hist)
    theta_loc   = ln_iv_hist.mean()
    sigma_loc   = ln_iv_hist.std()
    iv_stats_local[sid] = (theta_loc, sigma_loc)
    x_hist      = (theta_loc - ln_iv_hist) / sigma_loc
    dln_hist    = ln_iv_hist.diff()
    x_sh_all    = x_hist.shift(1).dropna()
    common_idx  = x_sh_all.index.intersection(dln_hist.dropna().index)
    xg, _, mm   = estimate_drift_function(x_sh_all.loc[common_idx].values,
                                           dln_hist.loc[common_idx].values)
    drift_functions_local[sid] = (xg, mm)

    mu_interp_loc = interp1d(xg, mm, kind='linear', fill_value='extrapolate')
    dln_aligned   = dln_hist.loc[x_sh_all.index].dropna()
    common2       = x_sh_all.index.intersection(dln_aligned.index)
    zeta_panel_local[sid] = dln_aligned.loc[common2] - pd.Series(
        mu_interp_loc(x_sh_all.loc[common2].values), index=common2)

hist_innovations = df[['epsilon']].join(zeta_panel_local, how='inner')
hist_innovations = hist_innovations[hist_innovations.index < test_date].dropna()
n_hist           = len(hist_innovations)
print(f"Historical observations for the bootstrap (no look-ahead): {n_hist}")

# Joint resampling
np.random.seed(42)
sampled_indices = np.random.randint(0, n_hist, size=(N_PATHS, HORIZON_DAYS))
eps_sampled     = hist_innovations['epsilon'].values[sampled_indices]
zeta_sampled    = {sid: hist_innovations[sid].values[sampled_indices]
                   for sid in grid_df['series_id']}

# Path propagation
price_paths = np.zeros((N_PATHS, HORIZON_DAYS + 1))
price_paths[:, 0] = F0
iv_paths = {sid: np.zeros((N_PATHS, HORIZON_DAYS + 1)) for sid in grid_df['series_id']}
for sid in grid_df['series_id']:
    iv_paths[sid][:, 0] = iv_at_test[sid]

for d in range(HORIZON_DAYS):
    I_d = iv_paths[ATM_SERIES][:, d]
    price_paths[:, d + 1] = price_paths[:, d] * np.exp(I_d * np.sqrt(dt) * eps_sampled[:, d])
    mean_p = price_paths[:, d + 1].mean()
    price_paths[:, d + 1] = (F0 / mean_p) * price_paths[:, d + 1]
    for sid in grid_df['series_id']:
        theta_s, sigma_s = iv_stats_local[sid]
        xg, mm           = drift_functions_local[sid]
        mu_interp_s      = interp1d(xg, mm, kind='linear', fill_value='extrapolate')
        V_d  = iv_paths[sid][:, d]
        x_d  = (theta_s - np.log(V_d)) / sigma_s
        mu_d = mu_interp_s(x_d)
        iv_paths[sid][:, d + 1] = V_d * np.exp(mu_d + zeta_sampled[sid][:, d])

atm_iv_paths = iv_paths[ATM_SERIES]
print(f"Price paths: final mean={price_paths[:,-1].mean():.4f}  skew={pd.Series(price_paths[:,-1]).skew():.4f}")

# Figure: path diagnostics
n_show = 200
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Bootstrapped Paths — Date: {test_date.strftime("%Y-%m-%d")}', fontsize=14)
for i in range(n_show):
    axes[0, 0].plot(range(HORIZON_DAYS + 1), price_paths[i, :],   alpha=0.1, lw=0.5, color='orange')
    axes[0, 1].plot(range(HORIZON_DAYS + 1), atm_iv_paths[i, :] * 100, alpha=0.1, lw=0.5, color='orange')
    axes[1, 0].plot(range(HORIZON_DAYS + 1), iv_paths['d10_30d'][i, :] * 100, alpha=0.1, lw=0.5, color='orange')
axes[0, 0].plot(range(HORIZON_DAYS + 1), price_paths.mean(axis=0), lw=2, color='black', label='Mean')
axes[0, 0].axhline(F0, color='red', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Price Paths')
axes[0, 0].set_ylabel('Normalized Price ($)')
axes[0, 0].legend()
axes[0, 1].plot(range(HORIZON_DAYS + 1), atm_iv_paths.mean(axis=0) * 100, lw=2, color='black', label='Mean')
axes[0, 1].set_title('IV ATM 30d Paths')
axes[0, 1].set_ylabel('IV ATM 30d (%)')
axes[0, 1].legend()
axes[1, 0].plot(range(HORIZON_DAYS + 1), iv_paths['d10_30d'].mean(axis=0) * 100, lw=2, color='black', label='Mean')
axes[1, 0].set_title('IV d10_30d Paths (deep OTM put)')
axes[1, 0].set_ylabel('IV (%)')
axes[1, 0].legend()
axes[1, 1].hist(price_paths[:, -1], bins=100, density=True, alpha=0.7, color='steelblue')
axes[1, 1].axvline(F0, color='red', linestyle='--', label=f'F0={F0}')
axes[1, 1].axvline(price_paths[:, -1].mean(), color='black', label=f"Mean={price_paths[:,-1].mean():.2f}")
axes[1, 1].set_xlabel('Final Price ($)')
axes[1, 1].set_title('Price Distribution at End of Week')
axes[1, 1].legend()
plt.tight_layout()
save_fig('fig_bootstrap_paths.png')

# ══════════════════════════════════════════════════════════════════════════════
# 9. DELTA-HEDGED EXCESS RETURNS + CONDITIONAL RISK PREMIUM
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 9. DELTA-HEDGED EXCESS RETURNS ===")

# Black-76 functions, discounted at the OptionMetrics rate r. The dividend yield is
# absorbed in the forward (Black-76), so no separate q term is needed.
def bms_call_price(F, K, iv, tau, r=0.0):
    if tau <= 1e-10: return np.maximum(F - K, 0.0)
    d1 = (np.log(F / K) + 0.5 * iv**2 * tau) / (iv * np.sqrt(tau))
    return np.exp(-r * tau) * (F * sp_norm.cdf(d1) - K * sp_norm.cdf(d1 - iv * np.sqrt(tau)))

def bms_put_price(F, K, iv, tau, r=0.0):
    if tau <= 1e-10: return np.maximum(K - F, 0.0)
    d1 = (np.log(F / K) + 0.5 * iv**2 * tau) / (iv * np.sqrt(tau))
    return np.exp(-r * tau) * (K * sp_norm.cdf(-(d1 - iv * np.sqrt(tau))) - F * sp_norm.cdf(-d1))

def bms_call_delta(F, K, iv, tau, r=0.0):
    if tau <= 1e-10: return np.where(F > K, 1.0, 0.0)
    d1 = (np.log(F / K) + 0.5 * iv**2 * tau) / (iv * np.sqrt(tau))
    return np.exp(-r * tau) * sp_norm.cdf(d1)

def bms_put_delta(F, K, iv, tau, r=0.0):
    if tau <= 1e-10: return np.where(F < K, -1.0, 0.0)
    d1 = (np.log(F / K) + 0.5 * iv**2 * tau) / (iv * np.sqrt(tau))
    return np.exp(-r * tau) * (sp_norm.cdf(d1) - 1.0)

def strike_from_call_delta(F, delta_target, iv, tau):
    d1 = sp_norm.ppf(delta_target)
    return F * np.exp(-d1 * iv * np.sqrt(tau) + 0.5 * iv**2 * tau)

def strike_from_put_delta(F, delta_target, iv, tau):
    d1 = sp_norm.ppf(delta_target + 1)
    return F * np.exp(-d1 * iv * np.sqrt(tau) + 0.5 * iv**2 * tau)

# Build the contracts — strikes are inverted from om_delta.
# Maturity convention: tau_init is in calendar years (days/365, as in OptionMetrics) and
# decays by dt = 1/252 per simulated trading day, i.e. 1 trading day = 365/252 = 1.448
# calendar days on average — the same equivalence used to annualize returns (252) and
# maturities (365).
contracts = []
for _, row in grid_df.iterrows():
    sid       = row['series_id']
    om_delta  = row['om_delta']
    wu_delta  = row['wu_delta']
    cp        = row['cp_flag']
    days      = row['days']
    tau       = days / 365.0
    iv_init   = iv_at_test[sid]
    r_init    = get_risk_free_rate(test_date, tau, yield_curve)
    if cp == 'C':
        K = strike_from_call_delta(F0, om_delta / 100.0, iv_init, tau)
    else:
        K = strike_from_put_delta(F0,  om_delta / 100.0, iv_init, tau)
    contracts.append({'series_id': sid, 'wu_delta': wu_delta, 'om_delta': om_delta,
                      'cp_flag': cp, 'days': days, 'tau_init': tau,
                      'iv_init': iv_init, 'r_init': r_init, 'strike': K})

contracts_df = pd.DataFrame(contracts)
print(contracts_df[['series_id', 'cp_flag', 'wu_delta', 'days', 'strike', 'iv_init', 'r_init']].to_string(index=False))

# Delta-hedged P&L: a dollar P&L per $100 of forward notional, not a percentage
# return. The return on the option premium (ret_premium) is also computed as an
# alternative scaled measure.
results_pnl = {}
results_ret_premium = {}
print(f"\n{'Contract':<18} {'Strike':>8} {'C0':>8} {'Mean PnL':>10} {'PnL std':>10} {'PnL ann.':>10}")
print("-" * 70)

for contract in contracts:
    sid        = contract['series_id']
    cp         = contract['cp_flag']
    K          = contract['strike']
    tau_init   = contract['tau_init']
    r          = contract['r_init']
    price_fn   = bms_call_price  if cp == 'C' else bms_put_price
    delta_fn   = bms_call_delta  if cp == 'C' else bms_put_delta
    iv_contract = iv_paths[sid]

    C0_vec    = price_fn(price_paths[:, 0], K, iv_contract[:, 0], tau_init, r)
    hedge_pnl = np.zeros(N_PATHS)

    for d in range(HORIZON_DAYS):
        tau_d   = max(tau_init - d * dt, 1e-10)
        delta_d = delta_fn(price_paths[:, d], K, iv_contract[:, d], tau_d, r)
        hedge_pnl += delta_d * (price_paths[:, d + 1] - price_paths[:, d])

    tau_final       = max(tau_init - HORIZON_DAYS * dt, 1e-10)
    C_final         = price_fn(price_paths[:, -1], K, iv_contract[:, -1], tau_final, r)
    pnl             = (C_final - C0_vec) - hedge_pnl
    results_pnl[sid]         = pnl
    results_ret_premium[sid] = pnl / C0_vec

    rp      = pnl.mean()
    rp_std  = pnl.std()
    rp_ann  = rp / (HORIZON_DAYS / 252)
    print(f"{sid:<18} {K:>8.2f} {C0_vec.mean():>8.4f} {rp:>10.4f} {rp_std:>10.4f} {rp_ann:>10.2f}")

results = results_pnl  # alias (dollar P&L) used by the figures below

# Table: conditional RP
rp_summary = []
for contract in contracts:
    sid = contract['series_id']
    er  = results_pnl[sid]
    rp     = er.mean()
    rp_std = er.std()
    rp_ann = rp / (HORIZON_DAYS / 252)
    sharpe = rp / rp_std if rp_std > 0 else 0
    rp_summary.append({
        'series_id':    sid,
        'wu_delta':     contract['wu_delta'],
        'cp_flag':      contract['cp_flag'],
        'days':         contract['days'],
        'rp_weekly':    rp,
        'rp_annualized': rp_ann,
        'rp_std':       rp_std,
        'ret_premium_mean': results_ret_premium[sid].mean(),
        'sharpe_ann':   sharpe * np.sqrt(252 / HORIZON_DAYS),
    })
rp_df = pd.DataFrame(rp_summary)
save_csv(rp_df.set_index('series_id'), 'tab_conditional_rp.csv')

# ── Realized delta-hedged P&L: forecast vs. actual market data ──
print("\n=== REALIZED DELTA-HEDGED P&L (market data, test_date + 5 trading days) ===")
trading_dates = df.index.sort_values()
future_dates  = trading_dates[trading_dates > test_date][:HORIZON_DAYS]
realized_dates = pd.DatetimeIndex([test_date]).append(future_dates)
spx_real_path  = df.loc[realized_dates, 'close'].values
F_real_path    = F0 * spx_real_path / spx_real_path[0]
n_real_days    = len(future_dates)

realized_rows = []
for contract in contracts:
    sid = contract['series_id']
    cp, K, tau_init, r = contract['cp_flag'], contract['strike'], contract['tau_init'], contract['r_init']
    price_fn, delta_fn = (bms_call_price, bms_call_delta) if cp == 'C' else (bms_put_price, bms_put_delta)

    iv_real = []
    for dte in realized_dates:
        if dte in iv_panel.index and pd.notna(iv_panel.loc[dte, sid]):
            iv_real.append(iv_panel.loc[dte, sid])
        else:
            iv_real.append(iv_panel[sid].loc[:dte].dropna().iloc[-1])
    iv_real = np.array(iv_real)

    C0_real = price_fn(F_real_path[0], K, iv_real[0], tau_init, r)
    hedge_pnl_real = 0.0
    for d in range(n_real_days):
        tau_d = max(tau_init - d * dt, 1e-10)
        delta_d = delta_fn(F_real_path[d], K, iv_real[d], tau_d, r)
        hedge_pnl_real += delta_d * (F_real_path[d + 1] - F_real_path[d])
    tau_final_r = max(tau_init - n_real_days * dt, 1e-10)
    C_final_real = price_fn(F_real_path[-1], K, iv_real[-1], tau_final_r, r)
    pnl_real = (C_final_real - C0_real) - hedge_pnl_real

    realized_rows.append({'series_id': sid, 'pnl_realized': pnl_real,
                           'ret_premium_realized': pnl_real / C0_real})

realized_df = pd.DataFrame(realized_rows).merge(
    rp_df[['series_id', 'rp_weekly', 'rp_std']], on='series_id')
realized_df['forecast_error'] = realized_df['pnl_realized'] - realized_df['rp_weekly']
realized_df['forecast_error_in_std'] = realized_df['forecast_error'] / realized_df['rp_std']
save_csv(realized_df.set_index('series_id'), 'tab_realized_vs_forecast.csv')
print("NOTE: N=1 test date -> a preliminary plausibility check, not a statistical")
print("test. The formal expectation-hypothesis tests over all 773 test dates are in")
print("Section 11 (pooled check) and Section 14 (cross-sectional and time-series).")

# Figure: RP by delta and maturity
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Conditional Risk Premium — {test_date.strftime("%Y-%m-%d")}', fontsize=14)
for days in SELECTED_MATURITIES:
    sub = rp_df[rp_df['days'] == days].sort_values('wu_delta')
    axes[0].plot(sub['wu_delta'], sub['rp_annualized'], marker='o', label=f'{days}d')
    axes[1].plot(sub['wu_delta'], sub['sharpe_ann'],    marker='s', label=f'{days}d')
for ax, ylabel, title in zip(axes,
    ['Annualized Risk Premium', 'Annualized Sharpe Ratio'],
    ['RP by Delta and Maturity', 'Sharpe Ratio by Delta and Maturity']):
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Put Delta (Wu convention)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
plt.tight_layout()
save_fig('fig_conditional_rp.png')

# Figure: distribution of delta-hedged P&L
show_contracts = ['d10_30d', 'd50_30d', 'd90_30d',
                  'd10_182d', 'd50_182d', 'd90_182d']
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Delta-Hedged Excess Return Distribution (Bootstrapped Paths)', fontsize=14)
for ax, sid in zip(axes.flat, show_contracts):
    if sid in results:
        er = results[sid]
        ax.hist(er, bins=80, density=True, alpha=0.7, color='steelblue')
        ax.axvline(0, color='red',   linestyle='--', alpha=0.5)
        ax.axvline(er.mean(), color='black', label=f'Mean={er.mean():.4f}')
        ax.set_title(sid)
        ax.legend(fontsize=8)
plt.tight_layout()
save_fig('fig_excess_return_distributions.png')

# ══════════════════════════════════════════════════════════════════════════════
# 10. 3D IV SURFACE IN MONEYNESS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 10. 3D IV SURFACE ===")

surface_points = []
for _, row in grid_df.iterrows():
    sid      = row['series_id']
    iv_mean  = iv_panel[sid].dropna().mean()
    tau      = row['days'] / 365.0
    om_delta = row['om_delta']
    cp       = row['cp_flag']
    if cp == 'C':
        K = strike_from_call_delta(F0, om_delta / 100.0, iv_mean, tau)
    else:
        K = strike_from_put_delta(F0,  om_delta / 100.0, iv_mean, tau)
    surface_points.append({
        'moneyness': K / F0,
        'maturity':  row['days'],
        'iv':        iv_mean * 100,
        'wu_delta':  row['wu_delta'],
        'series_id': sid,
    })
sp_df = pd.DataFrame(surface_points)

mon_fine = np.linspace(sp_df['moneyness'].min() - 0.02, sp_df['moneyness'].max() + 0.02, 60)
mat_fine = np.linspace(sp_df['maturity'].min(), sp_df['maturity'].max(), 60)
X_fine, Y_fine = np.meshgrid(mon_fine, mat_fine)
Z_fine = griddata(
    (sp_df['moneyness'].values, sp_df['maturity'].values),
    sp_df['iv'].values,
    (X_fine, Y_fine), method='cubic'
)

fig = plt.figure(figsize=(15, 9))
ax  = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_fine, Y_fine, Z_fine, cmap=cm.RdYlBu_r, alpha=0.9, edgecolor='none')
ax.scatter(sp_df['moneyness'], sp_df['maturity'], sp_df['iv'],
           color='black', s=25, zorder=5, label='Observed data')
ax.set_xlabel('Moneyness (K/S)', fontsize=12, labelpad=12)
ax.set_ylabel('Maturity (days)', fontsize=12, labelpad=12)
ax.set_zlabel('Implied Volatility (%)', fontsize=12, labelpad=10)
ax.set_title('S&P 500 Implied Volatility Surface\n(Sample mean, OptionMetrics)', fontsize=14)
ax.view_init(elev=25, azim=-130)
fig.colorbar(surf, shrink=0.5, aspect=15, label='IV (%)', pad=0.1)
ax.legend(fontsize=10)
plt.tight_layout()
save_fig('fig_iv_surface_3d.png')

# ── Skew view ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9))
ax  = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_fine, Y_fine, Z_fine, cmap=cm.RdYlBu_r, alpha=0.9, edgecolor='none')
ax.scatter(sp_df['moneyness'], sp_df['maturity'], sp_df['iv'], color='black', s=25, zorder=5)
ax.set_xlabel('Moneyness (K/S)', fontsize=12, labelpad=12)
ax.set_ylabel('Maturity (days)', fontsize=12, labelpad=12)
ax.set_zlabel('Implied Volatility (%)', fontsize=12, labelpad=10)
ax.set_title('S&P 500 IV Surface — Skew View', fontsize=14)
ax.view_init(elev=15, azim=-60)
fig.colorbar(surf, shrink=0.5, aspect=15, label='IV (%)', pad=0.1)
plt.tight_layout()
save_fig('fig_iv_surface_3d_skew.png')

# ══════════════════════════════════════════════════════════════════════════════
# 11. FULL ROLLING-WEDNESDAY LOOP (Wu 2025, sec. 3.2) — notebook Section 6
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 11. FULL ROLLING-WEDNESDAY LOOP ===")


def compute_zeta_panel(iv_panel, grid_df, estimate_drift_function, up_to_date=None,
                        bandwidth=7, n_grid=101, min_obs=500, verbose=True):
    """Rebuild the panel of zeta innovations using only observations up to and
    including up_to_date (no look-ahead). up_to_date=None uses the full sample,
    which is legitimate only for descriptive statistics."""
    drift_functions_ = {}
    iv_stats_ = {}
    zeta_panel_ = pd.DataFrame(index=iv_panel.index)
    for series_id in grid_df['series_id']:
        iv_full = iv_panel[series_id].dropna()
        iv_series = iv_full if up_to_date is None else iv_full.loc[:up_to_date]
        if len(iv_series) < min_obs:
            continue
        ln_iv = np.log(iv_series)
        theta_s = ln_iv.mean()
        sigma_s = ln_iv.std()
        x_s = (theta_s - ln_iv) / sigma_s
        dln_iv = ln_iv.diff()
        x_shifted = x_s.shift(1).dropna()
        common_idx = x_shifted.index.intersection(dln_iv.dropna().index)
        x_grid_s, _, mu_mono_s = estimate_drift_function(
            x_shifted.loc[common_idx].values, dln_iv.loc[common_idx].values,
            bandwidth=bandwidth, n_grid=n_grid)
        mu_interp_s = interp1d(x_grid_s, mu_mono_s, kind='linear', fill_value='extrapolate')
        x_all_shifted = x_s.shift(1).dropna()
        dln_aligned = dln_iv.loc[x_all_shifted.index].dropna()
        common = x_all_shifted.index.intersection(dln_aligned.index)
        zeta_s = dln_aligned.loc[common] - pd.Series(
            mu_interp_s(x_all_shifted.loc[common].values), index=common)
        drift_functions_[series_id] = (x_grid_s, mu_mono_s)
        iv_stats_[series_id] = (theta_s, sigma_s)
        zeta_panel_[series_id] = zeta_s
    return zeta_panel_, drift_functions_, iv_stats_


def run_bootstrap_for_date(test_date, horizon_days, n_paths, rng,
                            iv_panel, grid_df, df, yield_curve,
                            estimate_drift_function, compute_zeta_panel,
                            dt, F0, atm_series_id='d50_30d', min_hist=500):
    """Bootstrap the 25 contracts at test_date. Returns (forecast_dict, cov_matrix,
    realized_dict, series_ids), or None if the history available at test_date is
    insufficient."""
    zeta_panel_t, drift_functions_t, iv_stats_t = compute_zeta_panel(
        iv_panel, grid_df, estimate_drift_function, up_to_date=test_date,
        bandwidth=7, n_grid=101, verbose=False)

    eps_hist = df['epsilon'].loc[:test_date]
    common_idx = eps_hist.index.intersection(zeta_panel_t.dropna().index)
    common_idx = common_idx[common_idx < test_date]
    hist_innov = pd.DataFrame({'epsilon': eps_hist.loc[common_idx]})
    series_ids = list(grid_df['series_id'])
    for sid in series_ids:
        hist_innov[sid] = zeta_panel_t.loc[common_idx, sid]
    hist_innov = hist_innov.dropna()
    n_hist = len(hist_innov)
    if n_hist < min_hist:
        return None

    sampled_idx = rng.integers(0, n_hist, size=(n_paths, horizon_days))
    eps_s = hist_innov['epsilon'].values[sampled_idx]
    zeta_s = {sid: hist_innov[sid].values[sampled_idx] for sid in series_ids}

    iv_at_t = {}
    for sid in series_ids:
        if test_date in iv_panel.index and pd.notna(iv_panel.loc[test_date, sid]):
            iv_at_t[sid] = iv_panel.loc[test_date, sid]
        else:
            iv_at_t[sid] = iv_panel[sid].loc[:test_date].dropna().iloc[-1]

    price_p = np.zeros((n_paths, horizon_days + 1))
    price_p[:, 0] = F0
    iv_p = {sid: np.zeros((n_paths, horizon_days + 1)) for sid in series_ids}
    for sid in series_ids:
        iv_p[sid][:, 0] = iv_at_t[sid]

    mu_interp_cache = {sid: interp1d(*drift_functions_t[sid], kind='linear', fill_value='extrapolate')
                        for sid in series_ids}

    for d in range(horizon_days):
        I_d = iv_p[atm_series_id][:, d]
        price_p[:, d + 1] = price_p[:, d] * np.exp(I_d * np.sqrt(dt) * eps_s[:, d])
        mean_price = price_p[:, d + 1].mean()
        price_p[:, d + 1] = (F0 / mean_price) * price_p[:, d + 1]
        for sid in series_ids:
            theta_s, sigma_s = iv_stats_t[sid]
            V_d = iv_p[sid][:, d]
            x_d = (theta_s - np.log(V_d)) / sigma_s
            mu_d = mu_interp_cache[sid](x_d)
            iv_p[sid][:, d + 1] = V_d * np.exp(mu_d + zeta_s[sid][:, d])

    contracts_t = []
    for _, row in grid_df.iterrows():
        sid = row['series_id']
        cp = row['cp_flag']
        tau = row['days'] / 365.0
        iv_init = iv_at_t[sid]
        r_init = get_risk_free_rate(test_date, tau, yield_curve)
        delta_frac = row['om_delta'] / 100.0
        K = (strike_from_call_delta(F0, delta_frac, iv_init, tau) if cp == 'C'
             else strike_from_put_delta(F0, delta_frac, iv_init, tau))
        contracts_t.append({'series_id': sid, 'cp_flag': cp, 'tau_init': tau,
                             'iv_init': iv_init, 'r_init': r_init, 'strike': K})

    pnl_matrix = np.zeros((n_paths, len(contracts_t)))
    forecast = {}
    for ci, c in enumerate(contracts_t):
        sid = c['series_id']
        price_fn, delta_fn = (bms_call_price, bms_call_delta) if c['cp_flag'] == 'C' else (bms_put_price, bms_put_delta)
        iv_c = iv_p[sid]
        C0 = price_fn(price_p[:, 0], c['strike'], iv_c[:, 0], c['tau_init'], c['r_init'])
        hedge = np.zeros(n_paths)
        for d in range(horizon_days):
            tau_d = max(c['tau_init'] - d * dt, 1e-10)
            delta_d = delta_fn(price_p[:, d], c['strike'], iv_c[:, d], tau_d, c['r_init'])
            hedge += delta_d * (price_p[:, d + 1] - price_p[:, d])
        tau_f = max(c['tau_init'] - horizon_days * dt, 1e-10)
        C_f = price_fn(price_p[:, -1], c['strike'], iv_c[:, -1], tau_f, c['r_init'])
        pnl = (C_f - C0) - hedge
        pnl_matrix[:, ci] = pnl
        forecast[sid] = pnl.mean()

    cov_matrix = np.cov(pnl_matrix, rowvar=False)

    trading_dates = df.index.sort_values()
    future_dates = trading_dates[trading_dates > test_date][:horizon_days]
    realized_dates = pd.DatetimeIndex([test_date]).append(future_dates)
    spx_real = df.loc[realized_dates, 'close'].values
    F_real = F0 * spx_real / spx_real[0]

    realized = {}
    for c in contracts_t:
        sid = c['series_id']
        price_fn, delta_fn = (bms_call_price, bms_call_delta) if c['cp_flag'] == 'C' else (bms_put_price, bms_put_delta)
        iv_real = []
        for dte in realized_dates:
            if dte in iv_panel.index and pd.notna(iv_panel.loc[dte, sid]):
                iv_real.append(iv_panel.loc[dte, sid])
            else:
                iv_real.append(iv_panel[sid].loc[:dte].dropna().iloc[-1])
        iv_real = np.array(iv_real)
        C0_r = price_fn(F_real[0], c['strike'], iv_real[0], c['tau_init'], c['r_init'])
        hedge_r = 0.0
        for d in range(len(future_dates)):
            tau_d = max(c['tau_init'] - d * dt, 1e-10)
            delta_d = delta_fn(F_real[d], c['strike'], iv_real[d], tau_d, c['r_init'])
            hedge_r += delta_d * (F_real[d + 1] - F_real[d])
        tau_f_r = max(c['tau_init'] - len(future_dates) * dt, 1e-10)
        C_f_r = price_fn(F_real[-1], c['strike'], iv_real[-1], tau_f_r, c['r_init'])
        realized[sid] = (C_f_r - C0_r) - hedge_r

    return forecast, cov_matrix, realized, series_ids


# ── 11a. Test dates: every Wednesday from the sample midpoint ────────────────
all_dates_full = df.index
midpoint_full = all_dates_full[len(all_dates_full) // 2]
cal_wednesdays = pd.date_range(midpoint_full, all_dates_full.max(), freq='W-WED')

test_dates_list = []
_seen = set()
for w in cal_wednesdays:
    prior_trading = all_dates_full[all_dates_full <= w]
    if len(prior_trading) == 0:
        continue
    actual_date = prior_trading[-1]
    if actual_date in _seen:
        continue
    _seen.add(actual_date)
    test_dates_list.append(actual_date)
test_dates_full = pd.DatetimeIndex(sorted(test_dates_list))

_last_valid_idx = len(all_dates_full) - 1 - HORIZON_DAYS
_last_valid_date = all_dates_full[_last_valid_idx]
test_dates_full = test_dates_full[test_dates_full <= _last_valid_date]

print(f"Test dates: {len(test_dates_full)}  ({test_dates_full[0].date()} -- {test_dates_full[-1].date()})")

# ── 11b. Full loop ──────────────────────────────────────────────────────────
rng_rolling = np.random.default_rng(42)
t0_rolling = time.time()

rolling_records = []
rolling_cov = {}
n_ok, n_skipped, n_errors = 0, 0, 0

for i, td in enumerate(test_dates_full):
    try:
        out = run_bootstrap_for_date(
            td, HORIZON_DAYS, N_PATHS, rng_rolling,
            iv_panel, grid_df, df, yield_curve,
            estimate_drift_function, compute_zeta_panel, dt, F0,
        )
    except Exception as e:
        print(f"  [{i}] {td.date()}: EXCEPTION {type(e).__name__}: {e}")
        n_errors += 1
        continue
    if out is None:
        n_skipped += 1
        continue
    forecast, cov, realized, sids = out
    for sid in sids:
        rolling_records.append({
            'test_date': td, 'series_id': sid,
            'rp_forecast': forecast[sid],
            'pnl_realized': realized[sid],
            'forecast_error': realized[sid] - forecast[sid],
        })
    rolling_cov[td.strftime('%Y-%m-%d')] = cov
    n_ok += 1
    if (i + 1) % 200 == 0:
        elapsed = time.time() - t0_rolling
        rate = elapsed / (i + 1)
        remaining = rate * (len(test_dates_full) - i - 1)
        print(f"  [{i+1}/{len(test_dates_full)}]  elapsed={elapsed:.0f}s  est. remaining={remaining:.0f}s")

t_rolling_total = time.time() - t0_rolling
print(f"Done: {n_ok} ok, {n_skipped} skipped, {n_errors} errors, out of {len(test_dates_full)} in {t_rolling_total:.0f}s")

rolling_panel = pd.DataFrame.from_records(rolling_records)
print(f"Results panel: shape={rolling_panel.shape}")

# ── 11c. Save outputs ───────────────────────────────────────────────────────
panel_path = os.path.join(OUT, 'tab_rolling_conditional_rp.csv')
rolling_panel.to_csv(panel_path, index=False)   # long panel: the index carries no information
print(f"  → saved: tab_rolling_conditional_rp.csv  shape={rolling_panel.shape}")
cov_path = os.path.join(OUT, 'rolling_covariance_matrices.npz')
np.savez_compressed(cov_path, series_ids=np.array(list(grid_df['series_id'])), **rolling_cov)
print(f"  → saved: rolling_covariance_matrices.npz  ({len(rolling_cov)} dates)")

# ── 11d. Plausibility check: pooled expectation-hypothesis regression ──────
import statsmodels.api as sm

X_eh = sm.add_constant(rolling_panel['rp_forecast'])
y_eh = rolling_panel['pnl_realized']
model_eh = sm.OLS(y_eh, X_eh).fit(cov_type='HC1')

print("=== EXPECTATION HYPOTHESIS: realized = alpha + beta * forecast ===")
print(model_eh.summary().tables[1])
print(f"R-squared: {model_eh.rsquared:.4f}")

corr_by_series = rolling_panel.groupby('series_id').apply(
    lambda g: g['rp_forecast'].corr(g['pnl_realized']))
print("\nCorrelation (forecast, realized) by series:")
print(corr_by_series.sort_values(ascending=False).round(4).to_string())

# ── 11e. Figures ──────────────────────────────────────────────────────────────
# Figure: forecast vs. realized over time (ATM 30d, ATM 365d)
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle('Conditional Risk Premium: Forecast vs. Realized Over Time (ATM 30d and 365d)', fontsize=14)
for ax, sid, label in zip(axes, ['d50_30d', 'd50_365d'], ['ATM 30d', 'ATM 365d']):
    sub = rolling_panel[rolling_panel['series_id'] == sid].sort_values('test_date')
    ax.plot(sub['test_date'], sub['rp_forecast'], linewidth=0.8, color='black', label='Forecast (bootstrap)')
    ax.plot(sub['test_date'], sub['pnl_realized'], linewidth=0.5, alpha=0.6, color='darkred', label='Realized')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(label)
    ax.set_ylabel('P&L (USD per 100 notional)')
    ax.legend()
axes[-1].set_xlabel('Test date')
plt.tight_layout()
save_fig('fig_rolling_forecast_vs_realized.png')

# Figure: regression scatter
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(rolling_panel['rp_forecast'], rolling_panel['pnl_realized'], s=4, alpha=0.15, color='steelblue')
x_line = np.linspace(rolling_panel['rp_forecast'].min(), rolling_panel['rp_forecast'].max(), 100)
alpha_hat, beta_hat = model_eh.params['const'], model_eh.params['rp_forecast']
ax.plot(x_line, alpha_hat + beta_hat * x_line, color='darkred', linewidth=2,
        label=f'OLS fit: alpha={alpha_hat:.4f}, beta={beta_hat:.3f}')
ax.plot(x_line, x_line, color='black', linestyle='--', linewidth=1, label='45 degrees (H0: alpha=0, beta=1)')
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Forecast conditional risk premium (bootstrap)')
ax.set_ylabel('Realized delta-hedged P&L')
ax.set_title(f'Expectation Hypothesis Regression (pooled, N={len(rolling_panel)})\nR2 = {model_eh.rsquared:.4f}')
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
save_fig('fig_rolling_regression_scatter.png')

# Figure: correlation by delta x maturity
corr_grid = grid_df.copy()
corr_grid['corr'] = corr_grid['series_id'].map(corr_by_series)
corr_pivot = corr_grid.pivot(index='wu_delta', columns='days', values='corr')
corr_pivot = corr_pivot.reindex(index=[10, 25, 50, 75, 90], columns=SELECTED_MATURITIES)
fig, ax = plt.subplots(figsize=(8, 5))
vmax = np.abs(corr_pivot.values).max()
im = ax.imshow(corr_pivot.values, cmap=cm.RdYlBu_r, aspect='auto', vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(SELECTED_MATURITIES)))
ax.set_xticklabels([f'{d}d' for d in SELECTED_MATURITIES])
ax.set_yticks(range(5))
ax.set_yticklabels([10, 25, 50, 75, 90])
ax.set_xlabel('Maturity')
ax.set_ylabel('Put Delta (Wu convention)')
ax.set_title('Correlation (forecast, realized) by Delta x Maturity')
for i in range(5):
    for j in range(len(SELECTED_MATURITIES)):
        v = corr_pivot.values[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                color='white' if abs(v) > vmax * 0.6 else 'black')
plt.colorbar(im, ax=ax, label='Correlation')
plt.tight_layout()
save_fig('fig_rolling_correlation_by_contract.png')

# Figure: forecast-error distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(rolling_panel['forecast_error'], bins=150, density=True, alpha=0.7, color='steelblue')
ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero')
err_mean = rolling_panel['forecast_error'].mean()
err_std = rolling_panel['forecast_error'].std()
ax.axvline(err_mean, color='black', linewidth=2, label=f'Mean={err_mean:.4f}')
ax.set_xlabel('Forecast error (realized − forecast)')
ax.set_ylabel('Density')
ax.set_title(f"Distribution of the Forecast Error, Pooled Across All Contracts\n"
             f"(mean={err_mean:.4f}, std={err_std:.4f}, N={len(rolling_panel)})")
ax.legend()
plt.tight_layout()
save_fig('fig_rolling_forecast_error_dist.png')

# Figure: mean correlation structure across the 25 contracts
series_ids_ordered = list(grid_df['series_id'])
cov_stack = np.stack(list(rolling_cov.values()))
mean_cov = cov_stack.mean(axis=0)
std_diag = np.sqrt(np.diag(mean_cov))
mean_corr = mean_cov / np.outer(std_diag, std_diag)
fig, ax = plt.subplots(figsize=(11, 10))
im = ax.imshow(mean_corr, cmap=cm.RdYlBu_r, vmin=-1, vmax=1)
ax.set_xticks(range(len(series_ids_ordered)))
ax.set_xticklabels(series_ids_ordered, rotation=90, fontsize=7)
ax.set_yticks(range(len(series_ids_ordered)))
ax.set_yticklabels(series_ids_ordered, fontsize=7)
ax.set_title('Mean Delta-Hedged P&L Correlation Structure Across the 25 Contracts\n'
              '(average of the 773 conditional covariance matrices, normalized to correlation)')
plt.colorbar(im, ax=ax, label='Correlation', shrink=0.8)
plt.tight_layout()
save_fig('fig_rolling_covariance_heatmap.png')

# ══════════════════════════════════════════════════════════════════════════════
# 12. BOOTSTRAP PATH-PROPERTY DIAGNOSTIC — autocorrelation of simulated paths
# ══════════════════════════════════════════════════════════════════════════════
# Does the i.i.d. bootstrap still generate volatility clustering in simulated returns?
# Two design choices:
#  (a) the pooled ACF is computed with a PER-DAY demeaning (the cross-path mean at day d
#      is subtracted before pairing lags) rather than a single global mean. A global mean
#      confounds genuine within-path serial dependence with deterministic, ensemble-wide
#      effects (mean reversion toward theta from the initial state, and the per-step
#      renormalization below);
#  (b) the diagnostic is repeated from three initial states (low IV = last date, IV close
#      to the long-run mean theta, crisis IV = historical peak), since the answer depends
#      on the starting point.
# Note: the simulated return is NOT simply I_d*sqrt(dt)*eps_d -- the price cross-section
# is renormalized at every step so that its mean stays exactly F0
# (price_p[:, d+1] = (F0/mean_price) * ...), so the log increment is
# I_d*sqrt(dt)*eps_d + ln(F0/mean_price), a term common to all paths on day d. The
# per-day demeaning removes it too.
# Reuses the full-sample innovations_clean/drift_functions/iv_stats computed above.
# Diagnostic only: it does not touch N_PATHS/HORIZON_DAYS/seed of the main pipeline.
print("\n=== 12. BOOTSTRAP PATH-PROPERTY DIAGNOSTIC (three initial states, day-demeaned) ===")

N_PATHS_DIAG = 20_000
HORIZON_DIAG = 60
CHECK_SERIES_DIAG = ['d50_30d', 'd10_365d']


def pooled_acf_squared_global(panel, max_lag):
    """Global-demean estimator -- reported for comparison only, NOT the preferred
    estimate (see the note above)."""
    sq = panel ** 2
    return np.array([np.corrcoef(sq[:, :-k].ravel(), sq[:, k:].ravel())[0, 1]
                      for k in range(1, max_lag + 1)])


def pooled_acf_squared_daydemeaned(panel, max_lag):
    """Preferred estimator: subtracts the cross-path mean at day d BEFORE pairing
    lags, removing any component common to all paths on a given day (ensemble
    mean-reversion trend, renormalization adjustment) from the estimate of
    within-path serial dependence."""
    sq = panel ** 2
    dev = sq - sq.mean(axis=0, keepdims=True)
    out = []
    for k in range(1, max_lag + 1):
        a = dev[:, :-k].ravel()
        b = dev[:, k:].ravel()
        out.append((a * b).sum() / (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())))
    return np.array(out)


def run_path_diagnostic(init_label, iv_at_t, seed=42, max_lag=20):
    rng = np.random.default_rng(seed)
    n_hist = len(innovations_clean)
    sampled_idx = rng.integers(0, n_hist, size=(N_PATHS_DIAG, HORIZON_DIAG))
    eps_s = innovations_clean['epsilon'].values[sampled_idx]
    zeta_s = {sid: innovations_clean[sid].values[sampled_idx] for sid in grid_df['series_id']}
    price_p = np.zeros((N_PATHS_DIAG, HORIZON_DIAG + 1))
    price_p[:, 0] = F0
    iv_p = {sid: np.zeros((N_PATHS_DIAG, HORIZON_DIAG + 1)) for sid in grid_df['series_id']}
    for sid in grid_df['series_id']:
        iv_p[sid][:, 0] = iv_at_t[sid]
    mu_cache = {sid: interp1d(*drift_functions[sid], kind='linear', fill_value='extrapolate')
                for sid in grid_df['series_id']}
    for d in range(HORIZON_DIAG):
        I_d = iv_p[ATM_SERIES][:, d]
        price_p[:, d + 1] = price_p[:, d] * np.exp(I_d * np.sqrt(dt) * eps_s[:, d])
        mean_price = price_p[:, d + 1].mean()
        price_p[:, d + 1] = (F0 / mean_price) * price_p[:, d + 1]
        for sid in grid_df['series_id']:
            theta_s, sigma_s = iv_stats[sid]
            V_d = iv_p[sid][:, d]
            x_d = (theta_s - np.log(V_d)) / sigma_s
            mu_d = mu_cache[sid](x_d)
            iv_p[sid][:, d + 1] = V_d * np.exp(mu_d + zeta_s[sid][:, d])
    r_sim = np.diff(np.log(price_p), axis=1)
    acf_g = pooled_acf_squared_global(r_sim, max_lag)
    acf_d = pooled_acf_squared_daydemeaned(r_sim, max_lag)
    rows = []
    for k in range(max_lag):
        rows.append({'init_state': init_label, 'series': 'simulated_return', 'lag': k + 1,
                      'ac_squared_global': acf_g[k], 'ac_squared_daydemeaned': acf_d[k]})
    zeta_acf = {}
    for sid in CHECK_SERIES_DIAG:
        acf_zg = pooled_acf_squared_global(zeta_s[sid], max_lag)
        acf_zd = pooled_acf_squared_daydemeaned(zeta_s[sid], max_lag)
        zeta_acf[sid] = (acf_zg, acf_zd)
        for k in range(max_lag):
            rows.append({'init_state': init_label, 'series': f'simulated_zeta_{sid}', 'lag': k + 1,
                          'ac_squared_global': acf_zg[k], 'ac_squared_daydemeaned': acf_zd[k]})
    print(f"  [{init_label}] I0({ATM_SERIES})={iv_at_t[ATM_SERIES]:.4f}  "
          f"simulated return^2 AC(1): global={acf_g[0]:.4f}  day-demeaned={acf_d[0]:.4f}")
    return rows, acf_g, acf_d, zeta_acf


theta_atm, _ = iv_stats[ATM_SERIES]
target_level = np.exp(theta_atm)
s_atm = iv_panel[ATM_SERIES].dropna()
median_date = (s_atm - target_level).abs().idxmin()
crisis_date = s_atm.idxmax()

init_states = {
    'low_IV_last_date': {sid: iv_panel[sid].dropna().iloc[-1] for sid in grid_df['series_id']},
    'median_IV': {sid: (iv_panel[sid].dropna().loc[median_date] if median_date in iv_panel[sid].dropna().index
                         else iv_panel[sid].dropna().loc[:median_date].iloc[-1]) for sid in grid_df['series_id']},
    'crisis_IV': {sid: (iv_panel[sid].dropna().loc[crisis_date] if crisis_date in iv_panel[sid].dropna().index
                         else iv_panel[sid].dropna().loc[:crisis_date].iloc[-1]) for sid in grid_df['series_id']},
}
print(f"  Initial states: low_IV_last_date={iv_panel.index.max().date()}, "
      f"median_IV~theta date={median_date.date()}, crisis_IV date={crisis_date.date()}")

diag_results = []
diag_acf_by_state = {}
for label, iv_at_t in init_states.items():
    rows, acf_g, acf_d, zeta_acf = run_path_diagnostic(label, iv_at_t)
    diag_results.extend(rows)
    diag_acf_by_state[label] = (acf_g, acf_d, zeta_acf)

print(f"  (historical AC(1) s^2={acf_ret2[1]:.4f}, epsilon^2={acf_eps2[1]:.4f}, for qualitative comparison -- "
      f"NOT a calibration target: the historical statistic spans almost 30 years, the simulated one "
      f"a 60-day ensemble from a single initial state)")

diag_df = pd.DataFrame(diag_results)
diag_df.to_csv(os.path.join(OUT, 'tab_bootstrap_path_autocorrelation.csv'), index=False)
print("  → saved: tab_bootstrap_path_autocorrelation.csv")

max_lag_diag = 20
fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(0, color='gray', linewidth=0.8)
colors = {'low_IV_last_date': 'steelblue', 'median_IV': 'darkorange', 'crisis_IV': 'firebrick'}
width = 0.25
for i, (label, (acf_g, acf_d, _)) in enumerate(diag_acf_by_state.items()):
    ax.bar(np.arange(1, max_lag_diag + 1) + (i - 1) * width, acf_d, width=width,
           label=f'{label} (day-demeaned)', color=colors[label], alpha=0.85)
ax.axhline(acf_ret2[1], color='black', linestyle='--', linewidth=1,
           label=f'historical AC(1) $s^2$={acf_ret2[1]:.3f}')
ax.set_xlabel('Lag (trading days)')
ax.set_ylabel('Pooled, day-demeaned autocorrelation of squared simulated return')
ax.set_title(f'Bootstrap path-property diagnostic: pooled ACF by initial IV state\n'
             f'({N_PATHS_DIAG:,} simulated paths, {HORIZON_DIAG}-day horizon, diagnostic only)')
ax.legend(fontsize=8)
plt.tight_layout()
save_fig('fig_bootstrap_path_acf.png')

# ══════════════════════════════════════════════════════════════════════════════
# 13. CONDITIONAL RISK-PREMIUM SURFACE + REALIZED P&L SURFACE — with inference
# ══════════════════════════════════════════════════════════════════════════════
# Two distinct objects, reported separately:
#  13a. the conditional risk premium itself, i.e. the bootstrap forecast rp_forecast
#       (cross-path mean of the simulated P&L, known at the start of the week);
#  13b. the realized P&L observed afterwards, with statistical inference (HAC t-stat,
#       median, percentiles, fraction of negative weeks) -- with skewness up to 7 and
#       kurtosis close to 100, raw sample means are not automatically reliable.
print("\n=== 13. CONDITIONAL RISK-PREMIUM SURFACE + REALIZED P&L SURFACE ===")

series_order = list(grid_df['series_id'])

# ── 13a. Surface of the forecast conditional risk premium (rp_forecast) ──────
def _summarize_forecast(g):
    fc = g['rp_forecast']
    return pd.Series({
        'n': len(fc), 'mean': fc.mean(), 'std': fc.std(),
        'frac_negative': (fc < 0).mean(),
    })


forecast_summary = rolling_panel.groupby('series_id', group_keys=True)[['rp_forecast']].apply(
    lambda g: _summarize_forecast(g)).reset_index()
_extracted = forecast_summary['series_id'].str.extract(r'd(\d+)_(\d+)d').astype(int)
forecast_summary['wu_delta'], forecast_summary['days'] = _extracted[0], _extracted[1]
forecast_summary = forecast_summary.sort_values(['days', 'wu_delta']).reset_index(drop=True)

# Monte Carlo confidence interval for the simulation estimate of the conditional
# mean: hat(eta)_{j,t} +/- 1.96 * sqrt(Sigma_{jj,t}/N). This is a CI for the
# estimate, not a predictive interval for future P&L; contract-dates are split into
# significantly negative / significantly positive / indistinguishable from zero.
mcse_rows = []
for td, g in rolling_panel.groupby('test_date'):
    td_key = pd.Timestamp(td).strftime('%Y-%m-%d')
    if td_key not in rolling_cov:
        continue
    Sigma_t = rolling_cov[td_key]
    g = g.set_index('series_id').loc[series_order]
    mcse = np.sqrt(np.diag(Sigma_t) / N_PATHS)
    z = g['rp_forecast'].values / mcse
    for sid, m, zz in zip(series_order, mcse, z):
        mcse_rows.append({'series_id': sid, 'mcse': m, 'z': zz})
mcse_df = pd.DataFrame(mcse_rows)
mcse_df['sig_neg'] = mcse_df['z'] < -1.96
mcse_df['sig_pos'] = mcse_df['z'] > 1.96
mcse_by_sid = mcse_df.groupby('series_id').agg(
    mcse_mean=('mcse', 'mean'), mcse_max=('mcse', 'max'),
    frac_sig_neg=('sig_neg', 'mean'), frac_sig_pos=('sig_pos', 'mean'))
mcse_by_sid['frac_indistinguishable'] = 1 - mcse_by_sid['frac_sig_neg'] - mcse_by_sid['frac_sig_pos']
forecast_summary = forecast_summary.merge(mcse_by_sid, on='series_id', how='left')
save_csv(forecast_summary.set_index('series_id'), 'tab_rolling_forecast_surface.csv')
print("--- rp_forecast (the conditional risk premium, eta_j,t) ---")
print(forecast_summary[['series_id', 'n', 'mean', 'std', 'frac_negative',
                         'mcse_mean', 'frac_sig_neg', 'frac_sig_pos', 'frac_indistinguishable']]
      .round(4).to_string(index=False))
print(f"  Pooled (across all {len(mcse_df)} contract-dates), 95% Monte Carlo CI for the estimated "
      f"conditional mean: significantly negative {mcse_df['sig_neg'].mean()*100:.1f}%, "
      f"significantly positive {mcse_df['sig_pos'].mean()*100:.1f}%, "
      f"indistinguishable from zero {(1 - mcse_df['sig_neg'].mean() - mcse_df['sig_pos'].mean())*100:.1f}%")

fc_mean_pivot = forecast_summary.pivot(index='wu_delta', columns='days', values='mean')
fc_frac_pivot = forecast_summary.pivot(index='wu_delta', columns='days', values='frac_negative')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, piv, title, cmap, vkw in [
    (axes[0], fc_mean_pivot, 'Mean forecast $\\hat\\eta_{j,t}$ (long buyer)\n$100 notional forward',
     'RdBu', dict(vmin=-abs(fc_mean_pivot.values).max(), vmax=abs(fc_mean_pivot.values).max())),
    (axes[1], fc_frac_pivot, 'Fraction of test dates with $\\hat\\eta_{j,t}<0$',
     'RdYlBu_r', dict(vmin=0, vmax=1)),
]:
    im = ax.imshow(piv.values, cmap=cmap, aspect='auto', **vkw)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel('Maturity (days)'); ax.set_ylabel('Wu delta')
    ax.set_title(title)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f'{piv.values[i, j]:.2f}' if piv.values[i, j] < 1 else f'{piv.values[i, j]:.0f}',
                    ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle('Conditional risk-premium surface $\\hat\\eta_{j,t}$ (bootstrapped forecast), 773-date rolling sample',
             fontsize=13)
plt.tight_layout()
save_fig('fig_rolling_forecast_surface_heatmap.png')

# ── 13b. Realized P&L surface, with statistical inference ──────────────────
def _summarize_pnl(g):
    pnl = g['pnl_realized']
    X = np.ones(len(pnl))
    hac = sm.OLS(pnl.values, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    return pd.Series({
        'n': len(pnl), 'mean': pnl.mean(), 'std': pnl.std(),
        'skew': pnl.skew(), 'kurt': pnl.kurtosis(),
        'median': pnl.median(), 'p5': pnl.quantile(0.05), 'p95': pnl.quantile(0.95),
        'frac_negative': (pnl < 0).mean(),
        't_hac': hac.tvalues[0], 'p_hac': hac.pvalues[0],
        'sharpe_ann': pnl.mean() / pnl.std() * np.sqrt(252 / HORIZON_DAYS),
    })


pnl_summary = rolling_panel.groupby('series_id', group_keys=True)[['pnl_realized']].apply(
    lambda g: _summarize_pnl(g)).reset_index()
_extracted = pnl_summary['series_id'].str.extract(r'd(\d+)_(\d+)d').astype(int)
pnl_summary['wu_delta'], pnl_summary['days'] = _extracted[0], _extracted[1]
pnl_summary = pnl_summary.sort_values(['days', 'wu_delta']).reset_index(drop=True)
_, pnl_summary['q_hac'], _, _ = multipletests(pnl_summary['p_hac'], alpha=0.05, method='fdr_bh')
save_csv(pnl_summary.set_index('series_id'), 'tab_rolling_pnl_summary.csv')
print("\n--- pnl_realized (observed outcome, HAC inference, maxlags=4) ---")
print(pnl_summary[['series_id', 'n', 'mean', 'median', 't_hac', 'p_hac', 'q_hac', 'frac_negative', 'sharpe_ann']]
      .round(4).to_string(index=False))
n_sig_hac = (pnl_summary['p_hac'] < 0.05).sum()
n_sig_fdr = (pnl_summary['q_hac'] < 0.05).sum()
print(f"\n  Contracts with a significant mean at p<0.05 (HAC): {n_sig_hac}/25")
print(f"  Contracts significant after FDR correction (Benjamini-Hochberg, q<0.05): {n_sig_fdr}/25")

pnl_pivot = pnl_summary.pivot(index='wu_delta', columns='days', values='mean')
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(pnl_pivot.values, cmap='RdBu', aspect='auto',
                vmin=-abs(pnl_pivot.values).max(), vmax=abs(pnl_pivot.values).max())
ax.set_xticks(range(len(pnl_pivot.columns)))
ax.set_xticklabels(pnl_pivot.columns)
ax.set_yticks(range(len(pnl_pivot.index)))
ax.set_yticklabels(pnl_pivot.index)
ax.set_xlabel('Maturity (days)')
ax.set_ylabel('Wu delta')
ax.set_title('Mean realized delta-hedged P&L (long buyer)\nby delta x maturity, 773-date rolling sample')
for i in range(pnl_pivot.shape[0]):
    for j in range(pnl_pivot.shape[1]):
        ax.text(j, i, f'{pnl_pivot.values[i, j]:.3f}', ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=ax, label=r'Mean P&L (\$ per \$100 notional forward)')
plt.tight_layout()
save_fig('fig_rolling_pnl_surface_heatmap.png')

# ══════════════════════════════════════════════════════════════════════════════
# 14. EXPECTATION HYPOTHESIS (cross-sectional + time-series), RISK-PREMIUM
#    ATTRIBUTION, PORTFOLIO STRATEGIES  —  Wu (2025) eq. (22)-(23), (29), (30)-(33)
# ══════════════════════════════════════════════════════════════════════════════
# Uses ONLY the rolling-loop outputs (rolling_panel: rp_forecast/pnl_realized;
# rolling_cov: the 773 conditional covariance matrices) plus closed-form Black-76
# Greeks -- no new Monte Carlo simulation. Return-level risk attribution on the
# simulated paths (Wu eq. 24/28) is not implemented: it would require re-running the
# rolling loop while storing path-level P&L.
print("\n=== 14. EXPECTATION HYPOTHESIS, RISK-PREMIUM ATTRIBUTION, PORTFOLIO STRATEGIES ===")

# ── 14a. Closed-form Greeks by date and contract (dollar vega/gamma/volga/vanna) ──
# Rebuilds strike/iv_init/tau_init/r_init exactly as in run_bootstrap_for_date, but
# WITHOUT the Monte Carlo simulation (closed-form formulas evaluated at F0 on each date,
# ~773x25 analytic evaluations). Gamma/vega/vanna/volga are identical for calls and puts
# (they do not depend on the payoff's linear part removed by put-call parity), so
# cp_flag does not matter here.
greek_by_date = {}
for td in test_dates_full:
    td_key = td.strftime('%Y-%m-%d')
    if td_key not in rolling_cov:
        continue
    row_greeks = {}
    for _, row in grid_df.iterrows():
        sid = row['series_id']
        tau = row['days'] / 365.0
        if td in iv_panel.index and pd.notna(iv_panel.loc[td, sid]):
            iv_init = iv_panel.loc[td, sid]
        else:
            iv_init = iv_panel[sid].loc[:td].dropna().iloc[-1]
        r_init = get_risk_free_rate(td, tau, yield_curve)
        delta_frac = row['om_delta'] / 100.0
        K = (strike_from_call_delta(F0, delta_frac, iv_init, tau) if row['cp_flag'] == 'C'
             else strike_from_put_delta(F0, delta_frac, iv_init, tau))
        d1 = (np.log(F0 / K) + 0.5 * iv_init ** 2 * tau) / (iv_init * np.sqrt(tau))
        d2 = d1 - iv_init * np.sqrt(tau)
        disc = np.exp(-r_init * tau)
        n_d1 = norm.pdf(d1)
        dollar_vega = disc * F0 * n_d1 * np.sqrt(tau) * iv_init
        dollar_gamma = disc * F0 * n_d1 / (iv_init * np.sqrt(tau))
        dollar_volga = disc * F0 * n_d1 * np.sqrt(tau) * iv_init * d1 * d2
        dollar_vanna = -disc * F0 * n_d1 * d2
        row_greeks[sid] = np.array([dollar_vega, dollar_gamma, dollar_volga, dollar_vanna])
    greek_by_date[td_key] = row_greeks
print(f"  Greeks computed for {len(greek_by_date)} dates x 25 contracts (closed-form Black-76, "
      f"no additional simulation)")

series_order = list(grid_df['series_id'])

# ── 14b. Expectation hypothesis: cross-sectional (eq. 22) and time-series (eq. 23) regressions ──
print("\n--- 14b. Expectation hypothesis ---")

cs_rows = []
for td, g in rolling_panel.groupby('test_date'):
    g = g.set_index('series_id').loc[series_order]
    X = sm.add_constant(g['rp_forecast'].values)
    y = g['pnl_realized'].values
    if len(g) < 5:
        continue
    fit = sm.OLS(y, X).fit()
    cs_rows.append({'test_date': td, 'a_t': fit.params[0], 'b_t': fit.params[1], 'R2_t': fit.rsquared})
cs_df = pd.DataFrame(cs_rows)
save_csv(cs_df.set_index('test_date'), 'tab_eh_cross_sectional.csv')

a_fit = sm.OLS(cs_df['a_t'].values, np.ones(len(cs_df))).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
b_fit = sm.OLS(cs_df['b_t'].values - 1.0, np.ones(len(cs_df))).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(f"  Cross-sectional (eq. 22), {len(cs_df)} dates: "
      f"mean(a_t)={cs_df['a_t'].mean():.4f} (NW t vs 0: {a_fit.tvalues[0]:.2f}), "
      f"mean(b_t)={cs_df['b_t'].mean():.4f} (NW t vs 1: {b_fit.tvalues[0]:.2f}), "
      f"mean(R2_t)={cs_df['R2_t'].mean():.4f}")

ts_rows = []
for sid, g in rolling_panel.groupby('series_id'):
    X = sm.add_constant(g['rp_forecast'].values)
    y = g['pnl_realized'].values
    fit = sm.OLS(y, X).fit()
    ts_rows.append({'series_id': sid, 'a_k': fit.params[0], 'b_k': fit.params[1], 'R2_k': fit.rsquared})
ts_df = pd.DataFrame(ts_rows)
save_csv(ts_df.set_index('series_id'), 'tab_eh_time_series.csv')
print(f"  Time-series (eq. 23), 25 contracts: mean(a_k)={ts_df['a_k'].mean():.4f}, "
      f"mean(b_k)={ts_df['b_k'].mean():.4f} (std={ts_df['b_k'].std():.4f}), "
      f"mean(R2_k)={ts_df['R2_k'].mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(ts_df['a_k'], bins=12, color='steelblue', alpha=0.8)
axes[0].axvline(0, color='red', linestyle='--')
axes[0].set_title('Time-series intercept $a_k$ (25 contracts)')
axes[1].hist(ts_df['b_k'], bins=12, color='darkorange', alpha=0.8)
axes[1].axvline(1, color='red', linestyle='--')
axes[1].set_title('Time-series slope $b_k$ (25 contracts)')
plt.tight_layout()
save_fig('fig_eh_time_series_histograms.png')

# ── 14c. Risk-premium attribution (eq. 29): eta_t = X_t phi_t + e, per date, no intercept ──
print("\n--- 14c. Risk-premium attribution (eq. 29) ---")
phi_rows = []
for td, g in rolling_panel.groupby('test_date'):
    td_key = pd.Timestamp(td).strftime('%Y-%m-%d')
    if td_key not in greek_by_date:
        continue
    g = g.set_index('series_id').loc[series_order]
    Xt = np.vstack([greek_by_date[td_key][sid] for sid in series_order])  # 25x4
    eta_t = g['rp_forecast'].values
    phi_t, *_ = np.linalg.lstsq(Xt, eta_t, rcond=None)
    fitted = Xt @ phi_t
    pc = phi_t * (Xt.T @ Xt @ phi_t) / (eta_t @ eta_t) if eta_t @ eta_t > 0 else np.full(4, np.nan)
    phi_rows.append({'test_date': td, 'phi_vega': phi_t[0], 'phi_gamma': phi_t[1],
                      'phi_volga': phi_t[2], 'phi_vanna': phi_t[3],
                      'pc_vega': pc[0], 'pc_gamma': pc[1], 'pc_volga': pc[2], 'pc_vanna': pc[3]})
phi_df = pd.DataFrame(phi_rows)
save_csv(phi_df.set_index('test_date'), 'tab_risk_premium_attribution.csv')
print(f"  {len(phi_df)} dates. Mean phi (coefficient on each risk source): "
      f"vega={phi_df['phi_vega'].mean():.5f}  gamma={phi_df['phi_gamma'].mean():.5f}  "
      f"volga={phi_df['phi_volga'].mean():.5f}  vanna={phi_df['phi_vanna'].mean():.5f}")
print(f"  Mean contribution to the variance of the premium (pc, positive = explanatory): "
      f"vega={phi_df['pc_vega'].mean():.3f}  gamma={phi_df['pc_gamma'].mean():.3f}  "
      f"volga={phi_df['pc_volga'].mean():.3f}  vanna={phi_df['pc_vanna'].mean():.3f}")

fig, ax = plt.subplots(figsize=(9, 5))
pc_means = [phi_df['pc_vega'].mean(), phi_df['pc_gamma'].mean(), phi_df['pc_volga'].mean(), phi_df['pc_vanna'].mean()]
ax.bar(['Vega', 'Gamma', 'Volga', 'Vanna'], pc_means, color=['steelblue', 'darkorange', 'firebrick', 'seagreen'])
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_ylabel('Mean variance contribution to $\\hat\\eta_{j,t}$')
ax.set_title('Risk-premium attribution: mean contribution by risk source\n(773-date average, adapted from Wu (2025) eq. 29)')
plt.tight_layout()
save_fig('fig_risk_premium_attribution.png')

# ── 14d. Portfolio strategies (eq. 30-33): equal-notional, proportional, mean-variance, vega-neutral ──
print("\n--- 14d. Portfolio strategies ---")
port_returns = {k: [] for k in ['equal_notional', 'proportional', 'mean_variance', 'vega_neutral']}
port_dates = []
for td, g in rolling_panel.groupby('test_date'):
    td_key = pd.Timestamp(td).strftime('%Y-%m-%d')
    if td_key not in rolling_cov or td_key not in greek_by_date:
        continue
    g = g.set_index('series_id').loc[series_order]
    eta_t = g['rp_forecast'].values
    realized_t = g['pnl_realized'].values
    Sigma_t = rolling_cov[td_key]
    nu_t = np.array([greek_by_date[td_key][sid][0] for sid in series_order])  # dollar vega vector
    Sigma_inv = np.linalg.pinv(Sigma_t)

    w1 = np.full(25, 100.0)
    w2_raw = eta_t
    w3_raw = Sigma_inv @ eta_t
    denom = nu_t @ Sigma_inv @ nu_t
    proj = Sigma_inv @ (np.eye(25) - np.outer(nu_t, nu_t) @ Sigma_inv / denom) if denom != 0 else Sigma_inv
    w4_raw = proj @ eta_t

    def normalize(w_raw):
        s = np.abs(w_raw).sum()
        return w_raw * (2500.0 / s) if s > 0 else w_raw

    w2, w3, w4 = normalize(w2_raw), normalize(w3_raw), normalize(w4_raw)
    for key, w in [('equal_notional', w1), ('proportional', w2), ('mean_variance', w3), ('vega_neutral', w4)]:
        port_returns[key].append((w * realized_t / 100.0).sum())
    port_dates.append(td)

port_df = pd.DataFrame(port_returns, index=pd.DatetimeIndex(port_dates))
save_csv(port_df, 'tab_portfolio_strategies.csv')

perf_rows = []
ann = np.sqrt(252 / HORIZON_DAYS)
for key in port_returns:
    r = port_df[key]
    perf_rows.append({'strategy': key, 'mean_ann': r.mean() * (252 / HORIZON_DAYS),
                       'mean_weekly': r.mean(), 'vol_ann': r.std() * ann,
                       'information_ratio': (r.mean() / r.std()) * ann if r.std() > 0 else np.nan})
perf_df = pd.DataFrame(perf_rows)
save_csv(perf_df.set_index('strategy'), 'tab_portfolio_performance.csv')
print(perf_df[['strategy', 'mean_weekly', 'vol_ann', 'information_ratio']].round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
labels = ['Equal-notional', 'Proportional', 'Mean-variance', 'Vega-neutral']
irs = perf_df['information_ratio'].values
ax.bar(labels, irs, color=['gray', 'steelblue', 'darkorange', 'firebrick'])
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_ylabel('Annualized information ratio')
ax.set_title('Portfolio strategy performance, 773-date rolling sample\n(realized returns, $100 avg. gross notional per contract)')
plt.tight_layout()
save_fig('fig_portfolio_performance.png')

# ── 14e. Covariance benchmarks for the mean-variance strategy ───────────────
# Three alternative covariance inputs for the mean-variance weights:
#  (a) an "unconditional" full-sample average of all 773 Sigma_t, applied at every
#      date -- it uses future information, so it is an EX-POST ORACLE, not an
#      implementable strategy;
#  (b) a FEASIBLE past-only expanding-window average (Sigma_bar_t = mean of Sigma_s
#      for s < t), with a 52-date (~1-year) warmup before the strategy starts;
#  (c) diagonal only (no cross-contract correlation).
print("\n--- 14e. Covariance benchmarks (diagonal / ex-post oracle / feasible past-only) ---")

cov_dates = list(rolling_cov.keys())
cov_dates_sorted = sorted(pd.Timestamp(d) for d in cov_dates)
Sigma_oracle = np.mean([rolling_cov[d.strftime('%Y-%m-%d')] for d in cov_dates_sorted], axis=0)
Sigma_oracle_inv = np.linalg.pinv(Sigma_oracle)

COV_WARMUP = 52  # ~1 year of prior Wednesdays before the past-only strategy starts
cov_bench_returns = {'diagonal_only': [], 'oracle_full_sample': [], 'past_only_expanding': []}
cov_bench_dates = {'diagonal_only': [], 'oracle_full_sample': [], 'past_only_expanding': []}
running_sum = np.zeros_like(Sigma_oracle)
n_seen = 0
for td, g in rolling_panel.groupby('test_date'):
    td_key = pd.Timestamp(td).strftime('%Y-%m-%d')
    if td_key not in rolling_cov:
        continue
    g = g.set_index('series_id').loc[series_order]
    eta_t = g['rp_forecast'].values
    realized_t = g['pnl_realized'].values
    Sigma_t = rolling_cov[td_key]

    w_diag = normalize(eta_t / np.diag(Sigma_t))
    cov_bench_returns['diagonal_only'].append((w_diag * realized_t / 100.0).sum())
    cov_bench_dates['diagonal_only'].append(td)

    w_oracle = normalize(Sigma_oracle_inv @ eta_t)
    cov_bench_returns['oracle_full_sample'].append((w_oracle * realized_t / 100.0).sum())
    cov_bench_dates['oracle_full_sample'].append(td)

    if n_seen >= COV_WARMUP:
        Sigma_bar_inv = np.linalg.pinv(running_sum / n_seen)
        w_po = normalize(Sigma_bar_inv @ eta_t)
        cov_bench_returns['past_only_expanding'].append((w_po * realized_t / 100.0).sum())
        cov_bench_dates['past_only_expanding'].append(td)
    running_sum = running_sum + Sigma_t
    n_seen += 1

cov_bench_perf = []
for key in cov_bench_returns:
    r = pd.Series(cov_bench_returns[key], index=pd.DatetimeIndex(cov_bench_dates[key]))
    cov_bench_perf.append({'strategy': key, 'n_dates': len(r), 'mean_ann': r.mean() * (252 / HORIZON_DAYS),
                            'mean_weekly': r.mean(), 'vol_ann': r.std() * ann,
                            'information_ratio': (r.mean() / r.std()) * ann if r.std() > 0 else np.nan})
cov_bench_df = pd.DataFrame(cov_bench_perf)
save_csv(cov_bench_df.set_index('strategy'), 'tab_portfolio_covariance_benchmarks.csv')
print(f"  Warmup for past-only expanding benchmark: {COV_WARMUP} dates (~1 year of prior Wednesdays)")
print(cov_bench_df.round(4).to_string(index=False))
print("  NOTE: 'oracle_full_sample' uses the full 773-date average covariance at every date, "
      "including dates AFTER the test date for all but the last -- it is an ex-post diagnostic, "
      "not a feasible strategy. 'past_only_expanding' uses only Sigma_s for s<t and is feasible.")

# ── 14f. VRP^BTZ encompassing regression (H2 test) ───────────────────────────
# Per contract: realized P&L on eta and on the aggregate VRP^BTZ_t = VIX_t^2 - RV_t,
# with HAC standard errors and Benjamini-Hochberg FDR correction across contracts.
# (VRP^BTZ_t is constant across contracts on a date, so the test uses the time series.)
print("\n--- 14f. VRP^BTZ encompassing regression ---")

vrp_matched = df['VRP_seller'].reindex(test_dates_full)
enc_rows = []
for sid in series_order:
    g = rolling_panel[rolling_panel['series_id'] == sid].set_index('test_date').reindex(test_dates_full)
    y = g['pnl_realized'].values
    eta = g['rp_forecast'].values
    vrp = vrp_matched.values
    X = sm.add_constant(np.column_stack([eta, vrp]))
    fit = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    Xu = sm.add_constant(eta)
    fitu = sm.OLS(y, Xu, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    # VRP^BTZ-only regression, so that the incremental contribution can be read both ways
    # (eta-only vs. VRP-only vs. encompassing).
    Xv = sm.add_constant(vrp)
    fitv = sm.OLS(y, Xv, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    enc_rows.append({'series_id': sid, 'b_eta': fit.params[1], 't_eta': fit.tvalues[1], 'p_eta': fit.pvalues[1],
                      'c_vrp': fit.params[2], 't_vrp': fit.tvalues[2], 'p_vrp': fit.pvalues[2],
                      'R2_encompass': fit.rsquared, 'R2_univariate_eta': fitu.rsquared,
                      'R2_univariate_vrp': fitv.rsquared, 'b_vrp_only': fitv.params[1], 't_vrp_only': fitv.tvalues[1]})
enc_df = pd.DataFrame(enc_rows)
_, enc_df['q_eta'], _, _ = multipletests(enc_df['p_eta'], alpha=0.05, method='fdr_bh')
_, enc_df['q_vrp'], _, _ = multipletests(enc_df['p_vrp'], alpha=0.05, method='fdr_bh')
save_csv(enc_df.set_index('series_id'), 'tab_vrp_encompassing.csv')
print(f"  Mean R2: eta-only={enc_df['R2_univariate_eta'].mean():.4f}  vrp-only={enc_df['R2_univariate_vrp'].mean():.4f}  "
      f"encompassing={enc_df['R2_encompass'].mean():.4f}")
print(f"  |t_vrp|>1.96 (encompassing): {(enc_df['t_vrp'].abs() > 1.96).sum()}/25  |  |t_eta|>1.96: {(enc_df['t_eta'].abs() > 1.96).sum()}/25")
print(f"  |t_vrp_only|>1.96 (VRP alone): {(enc_df['t_vrp_only'].abs() > 1.96).sum()}/25")
print(f"  FDR q<0.05: eta {(enc_df['q_eta'] < 0.05).sum()}/25  |  vrp {(enc_df['q_vrp'] < 0.05).sum()}/25")

# ── 14g. Common-sample portfolio comparison + covariance conditioning/shrinkage ──
# (a) The feasible past-only benchmark (Section 14e) starts 52 dates later than the
# primary strategies, so all strategies are recomputed on the common post-warmup sample.
# (b) Numerical-stability check for the mean-variance weights: condition number, weight
# concentration, and sensitivity to ridge shrinkage of Sigma_t.
print("\n--- 14g. Common-sample portfolio comparison + covariance diagnostics ---")

common_rows = {k: [] for k in ['equal_notional', 'proportional', 'mean_variance', 'vega_neutral',
                                'diagonal_only', 'oracle_full_sample', 'past_only_expanding']}
n_seen_g = 0
running_sum_g = np.zeros_like(Sigma_oracle)
cond_numbers, min_eigs, max_weight_shares = [], [], []
shrink_lambdas = [0.01, 0.05, 0.10]
shrink_returns = {lam: [] for lam in shrink_lambdas}
unshrunk_returns = []

for td, g in rolling_panel.groupby('test_date'):
    td_key = pd.Timestamp(td).strftime('%Y-%m-%d')
    if td_key not in rolling_cov or td_key not in greek_by_date:
        continue
    g = g.set_index('series_id').loc[series_order]
    eta_t = g['rp_forecast'].values
    realized_t = g['pnl_realized'].values
    Sigma_t = rolling_cov[td_key]
    Sigma_inv = np.linalg.pinv(Sigma_t)
    nu_t = np.array([greek_by_date[td_key][sid][0] for sid in series_order])

    eigvals = np.linalg.eigvalsh(Sigma_t)
    min_eigs.append(eigvals.min())
    cond_numbers.append(eigvals.max() / eigvals.min() if eigvals.min() > 0 else np.inf)

    w3_raw = Sigma_inv @ eta_t
    w3 = normalize(w3_raw)
    max_weight_shares.append(np.abs(w3).max() / 2500.0)
    unshrunk_returns.append((w3 * realized_t / 100.0).sum())
    tr = np.trace(Sigma_t) / 25
    for lam in shrink_lambdas:
        Sigma_s = (1 - lam) * Sigma_t + lam * tr * np.eye(25)
        w_s = normalize(np.linalg.pinv(Sigma_s) @ eta_t)
        shrink_returns[lam].append((w_s * realized_t / 100.0).sum())

    if n_seen_g >= COV_WARMUP:
        Sigma_bar_inv = np.linalg.pinv(running_sum_g / n_seen_g)
        w_diag = normalize(eta_t / np.diag(Sigma_t))
        w_oracle = normalize(Sigma_oracle_inv @ eta_t)
        w_po = normalize(Sigma_bar_inv @ eta_t)
        denom = nu_t @ Sigma_inv @ nu_t
        proj = Sigma_inv @ (np.eye(25) - np.outer(nu_t, nu_t) @ Sigma_inv / denom) if denom != 0 else Sigma_inv
        w4 = normalize(proj @ eta_t)
        common_rows['equal_notional'].append((np.full(25, 100.0) * realized_t / 100.0).sum())
        common_rows['proportional'].append((normalize(eta_t) * realized_t / 100.0).sum())
        common_rows['mean_variance'].append((w3 * realized_t / 100.0).sum())
        common_rows['vega_neutral'].append((w4 * realized_t / 100.0).sum())
        common_rows['diagonal_only'].append((w_diag * realized_t / 100.0).sum())
        common_rows['oracle_full_sample'].append((w_oracle * realized_t / 100.0).sum())
        common_rows['past_only_expanding'].append((w_po * realized_t / 100.0).sum())
    running_sum_g = running_sum_g + Sigma_t
    n_seen_g += 1

common_perf = []
for key, vals in common_rows.items():
    r = pd.Series(vals)
    common_perf.append({'strategy': key, 'n_dates': len(r), 'mean_weekly': r.mean(), 'vol_ann': r.std() * ann,
                         'information_ratio': (r.mean() / r.std()) * ann if r.std() > 0 else np.nan})
common_perf_df = pd.DataFrame(common_perf)
save_csv(common_perf_df.set_index('strategy'), 'tab_portfolio_common_sample.csv')
print(f"  Common post-warmup sample: {len(common_rows['mean_variance'])} dates")
print(common_perf_df.round(4).to_string(index=False))

min_eigs = np.array(min_eigs); cond_numbers = np.array(cond_numbers); max_weight_shares = np.array(max_weight_shares)
cond_rows = [{'diagnostic': 'min_eigenvalue_min', 'value': min_eigs.min()},
             {'diagnostic': 'min_eigenvalue_median', 'value': np.median(min_eigs)},
             {'diagnostic': 'min_eigenvalue_max', 'value': min_eigs.max()},
             {'diagnostic': 'condition_number_min', 'value': cond_numbers.min()},
             {'diagnostic': 'condition_number_median', 'value': np.median(cond_numbers)},
             {'diagnostic': 'condition_number_max', 'value': cond_numbers.max()},
             {'diagnostic': 'n_non_psd', 'value': (min_eigs < 0).sum()},
             {'diagnostic': 'max_weight_share_mean', 'value': max_weight_shares.mean()},
             {'diagnostic': 'max_weight_share_median', 'value': np.median(max_weight_shares)},
             {'diagnostic': 'max_weight_share_max', 'value': max_weight_shares.max()}]
r0 = pd.Series(unshrunk_returns)
cond_rows.append({'diagnostic': 'mean_variance_IR_unshrunk', 'value': (r0.mean() / r0.std()) * ann})
for lam in shrink_lambdas:
    r = pd.Series(shrink_returns[lam])
    cond_rows.append({'diagnostic': f'mean_variance_IR_shrink_{lam}', 'value': (r.mean() / r.std()) * ann})
cond_df = pd.DataFrame(cond_rows)
save_csv(cond_df.set_index('diagnostic'), 'tab_covariance_diagnostics.csv')
print(f"\n  Condition number: median={np.median(cond_numbers):.1f}  range=[{cond_numbers.min():.1f}, {cond_numbers.max():.1f}]")
print(f"  Max |weight|/gross notional: mean={max_weight_shares.mean():.3f}  max={max_weight_shares.max():.3f}")
print(f"  Mean-variance IR: unshrunk={cond_rows[10]['value']:.4f}  "
      + "  ".join(f"shrink({lam})={cond_rows[11+i]['value']:.4f}" for i, lam in enumerate(shrink_lambdas)))

# ══════════════════════════════════════════════════════════════════════════════
# 15. ROBUSTNESS DIAGNOSTICS (bandwidth, alternative return scaling,
#     sub-samples, path-count sensitivity, HAC-lag choice, block bootstrap)
# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics only -- they do not change the defaults used everywhere else
# (bandwidth=7, IV-based scaling, N_PATHS=10000, i.i.d. resampling, maxlags=4).
# 15a/15f reuse the full-sample innovations_clean/drift_functions/iv_stats and
# init_states/median_date/crisis_date (Section 12) computed above.
print("\n=== 15. ROBUSTNESS DIAGNOSTICS ===")

# ── 15a. Bandwidth sensitivity, all 25 series (full sample) ────────────────
print("\n--- 15a. Bandwidth sensitivity (25 series, full sample) ---")

BANDWIDTH_GRID = [1.0, 2.0, 3.0, 5.0, 7.0]
bw_rows = []
for series_id in grid_df['series_id']:
    iv_series_bw = iv_panel[series_id].dropna()
    ln_iv_bw     = np.log(iv_series_bw)
    x_bw         = (ln_iv_bw.mean() - ln_iv_bw) / ln_iv_bw.std()
    dln_bw       = ln_iv_bw.diff()
    x_sh_bw      = x_bw.shift(1).dropna()
    common_bw    = x_sh_bw.index.intersection(dln_bw.dropna().index)
    x_vals_bw    = x_sh_bw.loc[common_bw].values
    y_vals_bw    = dln_bw.loc[common_bw].values
    for bw in BANDWIDTH_GRID:
        x_grid_bw, _, mu_mono_bw = estimate_drift_function(x_vals_bw, y_vals_bw, bandwidth=bw, n_grid=101)
        mu_interp_bw = interp1d(x_grid_bw, mu_mono_bw, kind='linear', fill_value='extrapolate')
        zeta_bw      = y_vals_bw - mu_interp_bw(x_vals_bw)
        i_m2 = np.argmin(np.abs(x_grid_bw - (-2)))
        i_p2 = np.argmin(np.abs(x_grid_bw - 2))
        bw_rows.append({'series_id': series_id, 'bandwidth': bw,
                         'mu_at_minus2': mu_mono_bw[i_m2], 'mu_at_plus2': mu_mono_bw[i_p2],
                         'ac1_zeta_sq': pd.Series(zeta_bw ** 2).autocorr(1), 'zeta_std': zeta_bw.std()})
bw_df = pd.DataFrame(bw_rows)
save_csv(bw_df, 'tab_bandwidth_sensitivity.csv')
print(f"  {len(grid_df)} series x {len(BANDWIDTH_GRID)} bandwidths = {len(bw_df)} estimations")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
BW_CHECK_SERIES = ['d10_30d', 'd50_30d', 'd90_30d', 'd10_365d', 'd50_365d', 'd90_365d']
for sid in BW_CHECK_SERIES:
    sub = bw_df[bw_df['series_id'] == sid].sort_values('bandwidth')
    axes[0].plot(sub['bandwidth'], sub['mu_at_plus2'] * 100, marker='o', label=sid)
    axes[1].plot(sub['bandwidth'], sub['ac1_zeta_sq'], marker='o', label=sid)
axes[0].set_xlabel('Bandwidth'); axes[0].set_ylabel(r'$\mu(x{=}+2)$, %')
axes[0].set_title('Drift at x=+2 vs. bandwidth'); axes[0].legend(fontsize=7)
axes[1].set_xlabel('Bandwidth'); axes[1].set_ylabel(r'AC(1) of $\zeta^2$')
axes[1].set_title('Residual AC(1) vs. bandwidth'); axes[1].legend(fontsize=7)
plt.tight_layout()
save_fig('fig_bandwidth_sensitivity.png')

# ── 15b. Alternative volatility measure for return scaling (RV vs. IV) ──
print("\n--- 15b. Alternative volatility measure for return scaling (realized vs. implied) ---")

df['epsilon_RV'] = df['log_return'] / (np.sqrt(df['RV_annual'].clip(lower=1e-8)).shift(1) * np.sqrt(dt))
eps_rv_clean  = df['epsilon_RV'].dropna()
eps_iv_common = df.loc[eps_rv_clean.index, 'epsilon']

def _innov_row(s):
    lb_stat, lb_pvalue = lb_q20_sq(s)
    return {'N': int(s.notna().sum()), 'mean': s.mean(), 'std': s.std(),
            'skew': s.skew(), 'kurtosis': s.kurtosis(), 'AC1_sq': ac1_sq(s),
            'LB_Q20_sq': lb_stat, 'LB_Q20_sq_pvalue': lb_pvalue}

alt_scaling_df = pd.DataFrame({'epsilon_IV (production)': _innov_row(eps_iv_common),
                                'epsilon_RV (alternative)': _innov_row(eps_rv_clean)}).T
save_csv(alt_scaling_df, 'tab_alternative_return_scaling.csv')
print(alt_scaling_df.round(4).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
lags_plot = np.arange(1, 21)
ax.plot(lags_plot, acf(eps_iv_common ** 2, nlags=20)[1:], marker='o', label=r'$\varepsilon_{IV}^2$ (production)')
ax.plot(lags_plot, acf(eps_rv_clean ** 2, nlags=20)[1:], marker='s', label=r'$\varepsilon_{RV}^2$ (alternative)')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Lag'); ax.set_ylabel('ACF'); ax.legend()
ax.set_title('Return-innovation scaling: implied vs. realized volatility')
plt.tight_layout()
save_fig('fig_alternative_return_scaling.png')

# ── 15c. Sub-sample analysis (first vs. second half of the 773 test dates) ──
print("\n--- 15c. Sub-sample analysis (first vs. second half of the rolling test dates) ---")

test_dates_sorted = sorted(rolling_panel['test_date'].unique())
split_date = test_dates_sorted[len(test_dates_sorted) // 2]
first_half  = rolling_panel[rolling_panel['test_date'] < split_date]
second_half = rolling_panel[rolling_panel['test_date'] >= split_date]

def _pooled_eh(sub):
    from scipy.stats import t as t_dist
    X = sm.add_constant(sub['rp_forecast'])
    m = sm.OLS(sub['pnl_realized'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    beta_se = m.bse['rp_forecast']
    t_vs_1 = (m.params['rp_forecast'] - 1.0) / beta_se
    p_vs_1 = 2 * (1 - t_dist.cdf(np.abs(t_vs_1), df=m.df_resid))
    return {'n_dates': sub['test_date'].nunique(), 'n_obs': len(sub),
            'date_start': sub['test_date'].min().date(), 'date_end': sub['test_date'].max().date(),
            'alpha': m.params['const'], 'beta': m.params['rp_forecast'], 'beta_se': beta_se,
            't_stat_vs_1': t_vs_1, 'p_value_vs_1': p_vs_1, 'R2': m.rsquared,
            'mean_rp_forecast': sub['rp_forecast'].mean(), 'mean_pnl_realized': sub['pnl_realized'].mean()}

subsample_df = pd.DataFrame({'full_sample': _pooled_eh(rolling_panel),
                              'first_half': _pooled_eh(first_half),
                              'second_half': _pooled_eh(second_half)}).T
save_csv(subsample_df, 'tab_subsample_analysis.csv')
print(subsample_df.round(4).to_string())

port_first  = port_df[port_df.index < split_date]
port_second = port_df[port_df.index >= split_date]
subsample_ir_rows = []
for key in port_returns:
    for label, sub in [('first_half', port_first), ('second_half', port_second)]:
        r = sub[key]
        subsample_ir_rows.append({'strategy': key, 'sub_sample': label,
                                   'information_ratio': (r.mean() / r.std()) * ann if r.std() > 0 else np.nan})
subsample_ir_df = pd.DataFrame(subsample_ir_rows)
save_csv(subsample_ir_df, 'tab_subsample_portfolio_ir.csv')
print(subsample_ir_df.round(4).to_string(index=False))

# ── 15d. Sensitivity to the number of bootstrap paths (N) ────────────────────────
print("\n--- 15d. Sensitivity to the number of bootstrap paths (N) ---")

atm_at_test = pd.Series({td: (iv_panel.loc[td, ATM_SERIES] if td in iv_panel.index and pd.notna(iv_panel.loc[td, ATM_SERIES])
                               else iv_panel[ATM_SERIES].loc[:td].dropna().iloc[-1]) for td in test_dates_full})
npaths_test_dates = {'low_IV': atm_at_test.idxmin(),
                      'median_IV': (atm_at_test - target_level).abs().idxmin(),
                      'crisis_IV': atm_at_test.idxmax()}
N_GRID = [1000, 2500, 5000, 10000, 20000]
CHECK_CONTRACTS_N = ['d10_30d', 'd50_30d', 'd90_30d', 'd50_365d']

npaths_rows = []
for label, td in npaths_test_dates.items():
    for n in N_GRID:
        out = run_bootstrap_for_date(td, HORIZON_DAYS, n, np.random.default_rng(42),
                                      iv_panel, grid_df, df, yield_curve,
                                      estimate_drift_function, compute_zeta_panel, dt, F0)
        if out is None:
            continue
        forecast_n, cov_n, realized_n, sids_n = out
        for sid in CHECK_CONTRACTS_N:
            i_sid = sids_n.index(sid)
            npaths_rows.append({'init_state': label, 'test_date': td, 'n_paths': n, 'series_id': sid,
                                 'eta_forecast': forecast_n[sid], 'mc_se': np.sqrt(cov_n[i_sid, i_sid] / n)})
npaths_df = pd.DataFrame(npaths_rows)
npaths_df['eta_at_10000'] = npaths_df.groupby(['init_state', 'series_id'])['eta_forecast'].transform(
    lambda s: s[npaths_df.loc[s.index, 'n_paths'] == 10000].iloc[0] if (npaths_df.loc[s.index, 'n_paths'] == 10000).any() else np.nan)
npaths_df['mc_se_at_10000'] = npaths_df.groupby(['init_state', 'series_id'])['mc_se'].transform(
    lambda s: s[npaths_df.loc[s.index, 'n_paths'] == 10000].iloc[0] if (npaths_df.loc[s.index, 'n_paths'] == 10000).any() else np.nan)
# z-score, not %-deviation, is the headline diagnostic: several contracts have eta_hat close to
# zero at some initial states (e.g. d50_30d at the median-IV state), where a %-deviation blows up
# even though the absolute deviation is well within Monte Carlo noise -- kept in the saved table
# for completeness but not used as the summary statistic below.
npaths_df['pct_dev_from_10000'] = (npaths_df['eta_forecast'] - npaths_df['eta_at_10000']) / npaths_df['eta_at_10000'].abs() * 100
npaths_df['z_vs_10000'] = ((npaths_df['eta_forecast'] - npaths_df['eta_at_10000'])
                            / np.sqrt(npaths_df['mc_se'] ** 2 + npaths_df['mc_se_at_10000'] ** 2))
save_csv(npaths_df, 'tab_npaths_sensitivity.csv')
print(f"  Test dates: {', '.join(f'{k}={v.date()}' for k, v in npaths_test_dates.items())}")
_z_nonbase = npaths_df.loc[npaths_df['n_paths'] != 10000, 'z_vs_10000']
print(f"  Max |z-score| vs. the N=10,000 baseline (N != 10,000 rows only): {_z_nonbase.abs().max():.2f} "
      f"({(_z_nonbase.abs() > 2).sum()}/{len(_z_nonbase)} exceed |z|=2) -- consistent with pure "
      f"Monte Carlo noise, not a systematic path-count effect")

fig, ax = plt.subplots(figsize=(9, 5))
for label in npaths_test_dates:
    sub = npaths_df[(npaths_df['init_state'] == label) & (npaths_df['series_id'] == 'd50_30d')].sort_values('n_paths')
    ax.errorbar(sub['n_paths'], sub['eta_forecast'], yerr=1.96 * sub['mc_se'], marker='o', capsize=3, label=label)
ax.set_xscale('log')
ax.set_xlabel('N (bootstrap paths)'); ax.set_ylabel(r'$\hat\eta_H$ (d50_30d), \$ per \$100 notional')
ax.set_title('Sensitivity of the forecast conditional risk premium to path count\n(error bars: 95% Monte Carlo interval)')
ax.legend(fontsize=8)
plt.tight_layout()
save_fig('fig_npaths_sensitivity.png')

# ── 15e. HAC-lag robustness of realized-P&L significance ──
print("\n--- 15e. HAC-lag robustness for realized-P&L significance ---")

LAG_GRID = [0, 1, 2, 4, 8, 12]
hac_lag_rows = []
for lag in LAG_GRID:
    for sid in series_order:
        sub = rolling_panel.loc[rolling_panel['series_id'] == sid, 'pnl_realized'].values
        X = np.ones(len(sub))
        m = sm.OLS(sub, X).fit() if lag == 0 else sm.OLS(sub, X).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
        hac_lag_rows.append({'maxlags': lag, 'series_id': sid, 'mean': sub.mean(),
                              't_stat': m.tvalues[0], 'p_value': m.pvalues[0]})
hac_lag_df = pd.DataFrame(hac_lag_rows)
save_csv(hac_lag_df, 'tab_hac_lag_robustness.csv')
hac_lag_summary = (hac_lag_df.groupby('maxlags')['p_value']
                    .apply(lambda p: (p < 0.05).sum()).rename('n_significant_of_25').reset_index())

# Stable counts do not imply stable contract identities: also track, relative to the production
# 4-lag choice, how many contracts are added/removed from the significant set at each other lag.
BASELINE_LAG = 4
sig_sets = {lag: set(hac_lag_df.loc[(hac_lag_df['maxlags'] == lag) & (hac_lag_df['p_value'] < 0.05),
                                     'series_id']) for lag in LAG_GRID}
baseline_set = sig_sets[BASELINE_LAG]
hac_lag_summary['n_added_vs_lag4']   = hac_lag_summary['maxlags'].map(lambda lag: len(sig_sets[lag] - baseline_set))
hac_lag_summary['n_removed_vs_lag4'] = hac_lag_summary['maxlags'].map(lambda lag: len(baseline_set - sig_sets[lag]))
hac_lag_summary['added_series_vs_lag4'] = hac_lag_summary['maxlags'].map(
    lambda lag: ','.join(sorted(sig_sets[lag] - baseline_set)))
save_csv(hac_lag_summary, 'tab_hac_lag_summary.csv')
print(hac_lag_summary.to_string(index=False))
print(f"  Baseline (maxlags={BASELINE_LAG}) significant set is a strict subset of every other lag's "
      f"set tested: {all(baseline_set <= s for s in sig_sets.values())}")

# ── 15f. Block bootstrap vs. i.i.d. (3 initial states, all 25 contracts) ──
print("\n--- 15f. Block bootstrap vs. i.i.d. resampling (3 initial states, 25 contracts) ---")

def sample_block_indices(n_hist, n_paths, horizon_days, block_len, rng):
    n_blocks = int(np.ceil(horizon_days / block_len))
    starts = rng.integers(0, max(n_hist - block_len, 1) + 1, size=(n_paths, n_blocks))
    idx = np.zeros((n_paths, n_blocks * block_len), dtype=int)
    for b in range(n_blocks):
        for offset in range(block_len):
            idx[:, b * block_len + offset] = np.minimum(starts[:, b] + offset, n_hist - 1)
    return idx[:, :horizon_days]

def propagate_and_price(iv_at_t, sampled_idx, n_paths_b, r_date):
    eps_s  = innovations_clean['epsilon'].values[sampled_idx]
    zeta_s = {sid: innovations_clean[sid].values[sampled_idx] for sid in grid_df['series_id']}
    price_p = np.zeros((n_paths_b, HORIZON_DAYS + 1)); price_p[:, 0] = F0
    iv_p = {sid: np.zeros((n_paths_b, HORIZON_DAYS + 1)) for sid in grid_df['series_id']}
    for sid in grid_df['series_id']:
        iv_p[sid][:, 0] = iv_at_t[sid]
    mu_cache_b = {sid: interp1d(*drift_functions[sid], kind='linear', fill_value='extrapolate')
                  for sid in grid_df['series_id']}
    for d in range(HORIZON_DAYS):
        I_d = iv_p[ATM_SERIES][:, d]
        price_p[:, d + 1] = price_p[:, d] * np.exp(I_d * np.sqrt(dt) * eps_s[:, d])
        price_p[:, d + 1] = (F0 / price_p[:, d + 1].mean()) * price_p[:, d + 1]
        for sid in grid_df['series_id']:
            theta_s, sigma_s = iv_stats[sid]
            V_d  = iv_p[sid][:, d]
            x_d  = (theta_s - np.log(V_d)) / sigma_s
            iv_p[sid][:, d + 1] = V_d * np.exp(mu_cache_b[sid](x_d) + zeta_s[sid][:, d])
    pnl_matrix_b = np.zeros((n_paths_b, len(grid_df)))
    for ci, row in grid_df.iterrows():
        sid, cp, tau0 = row['series_id'], row['cp_flag'], row['days'] / 365.0
        iv_init  = iv_at_t[sid]
        r_init   = get_risk_free_rate(r_date, tau0, yield_curve)
        delta_frac = row['om_delta'] / 100.0
        price_fn, delta_fn = (bms_call_price, bms_call_delta) if cp == 'C' else (bms_put_price, bms_put_delta)
        K = (strike_from_call_delta(F0, delta_frac, iv_init, tau0) if cp == 'C'
             else strike_from_put_delta(F0, delta_frac, iv_init, tau0))
        iv_c = iv_p[sid]
        C0 = price_fn(price_p[:, 0], K, iv_c[:, 0], tau0, r_init)
        hedge = np.zeros(n_paths_b)
        for d in range(HORIZON_DAYS):
            tau_d = max(tau0 - d * dt, 1e-10)
            hedge += delta_fn(price_p[:, d], K, iv_c[:, d], tau_d, r_init) * (price_p[:, d + 1] - price_p[:, d])
        tau_f = max(tau0 - HORIZON_DAYS * dt, 1e-10)
        C_f = price_fn(price_p[:, -1], K, iv_c[:, -1], tau_f, r_init)
        pnl_matrix_b[:, ci] = (C_f - C0) - hedge
    return pnl_matrix_b

BLOCK_LEN = HORIZON_DAYS
init_state_dates = {'low_IV_last_date': iv_panel.index.max(), 'median_IV': median_date, 'crisis_IV': crisis_date}
rng_block_diag = np.random.default_rng(42)
n_hist_full = len(innovations_clean)

block_rows = []
for label, iv_at_t in init_states.items():
    r_date = init_state_dates[label]
    idx_iid   = rng_block_diag.integers(0, n_hist_full, size=(N_PATHS, HORIZON_DAYS))
    idx_block = sample_block_indices(n_hist_full, N_PATHS, HORIZON_DAYS, BLOCK_LEN, rng_block_diag)
    pnl_iid   = propagate_and_price(iv_at_t, idx_iid,   N_PATHS, r_date)
    pnl_block = propagate_and_price(iv_at_t, idx_block, N_PATHS, r_date)
    for ci, row in grid_df.iterrows():
        block_rows.append({'init_state': label, 'series_id': row['series_id'],
                            'eta_iid': pnl_iid[:, ci].mean(), 'eta_block': pnl_block[:, ci].mean(),
                            'std_iid': pnl_iid[:, ci].std(), 'std_block': pnl_block[:, ci].std()})
block_df = pd.DataFrame(block_rows)
block_df['std_ratio_block_over_iid'] = block_df['std_block'] / block_df['std_iid']
# idx_iid and idx_block are two independent draws (sequential calls on the same rng stream,
# not common random numbers), so a naive point-estimate difference conflates the resampling-
# scheme effect with ordinary Monte Carlo variability between two runs. A z-score against each
# side's own Monte Carlo SE (std/sqrt(N)) is the honest headline statistic; the raw difference
# and %-difference are kept in the table but are not the summary figure below.
block_df['se_iid']   = block_df['std_iid'] / np.sqrt(N_PATHS)
block_df['se_block'] = block_df['std_block'] / np.sqrt(N_PATHS)
block_df['z_block_vs_iid'] = ((block_df['eta_block'] - block_df['eta_iid'])
                               / np.sqrt(block_df['se_iid'] ** 2 + block_df['se_block'] ** 2))
save_csv(block_df, 'tab_block_bootstrap_comparison.csv')
print(f"  Mean std ratio (block/i.i.d.), pooled across 25 contracts x 3 states: "
      f"{block_df['std_ratio_block_over_iid'].mean():.3f}")
print(f"  Mean |eta_block - eta_iid|: {np.abs(block_df['eta_block'] - block_df['eta_iid']).mean():.4f}")
print(f"  |z-score| (block vs. i.i.d., own Monte Carlo SEs) > 2: "
      f"{(block_df['z_block_vs_iid'].abs() > 2).sum()}/{len(block_df)} contract-states "
      f"(median |z|={block_df['z_block_vs_iid'].abs().median():.2f}, max |z|={block_df['z_block_vs_iid'].abs().max():.2f})")

fig, ax = plt.subplots(figsize=(6.5, 6.5))
colors_b = {'low_IV_last_date': 'steelblue', 'median_IV': 'darkorange', 'crisis_IV': 'firebrick'}
for label in init_states:
    sub = block_df[block_df['init_state'] == label]
    ax.scatter(sub['eta_iid'], sub['eta_block'], label=label, color=colors_b[label], alpha=0.75)
lims = [block_df[['eta_iid', 'eta_block']].min().min(), block_df[['eta_iid', 'eta_block']].max().max()]
ax.plot(lims, lims, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel(r'$\hat\eta_H$, i.i.d. resampling'); ax.set_ylabel(r'$\hat\eta_H$, block resampling (5-day blocks)')
ax.set_title('Block bootstrap vs. i.i.d.: forecast conditional risk premium\n(25 contracts x 3 initial states)')
ax.legend(fontsize=8)
plt.tight_layout()
save_fig('fig_block_bootstrap_comparison.png')

print("\n✓ All outputs saved to output/")
