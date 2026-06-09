import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Nexus · Retail Intelligence",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)


# ──────────────────────────────────────────────────────────────────
# THEME CONFIGURATION
# ──────────────────────────────────────────────────────────────────
theme_choice = st.sidebar.radio('Theme Mode', ['Dark', 'Light'], horizontal=True)
is_dark = (theme_choice == 'Dark')

C_BG      = '#0A0A0F' if is_dark else '#F8FAFC'
C_CARD    = '#131322' if is_dark else '#FFFFFF'
C_SIDEBAR = '#0F0F1A' if is_dark else '#F1F5F9'
C_TEXT    = '#EDEDF2' if is_dark else '#1E293B'
C_SUB     = '#A0A0B4' if is_dark else '#64748B'
C_BORD    = '#1E1E2A' if is_dark else '#E2E8F0'
C_HOVER   = '#1A1A28' if is_dark else '#E2E8F0'
C_ACCENT  = '#5B5BD6'
C_HIST    = '#2A2A40' if is_dark else '#94A3B8'
P_THEME   = 'plotly_dark' if is_dark else 'plotly_white'

# ──────────────────────────────────────────────────────────────────
# DARK THEME CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    /* ── ROOT DARK CANVAS ── */
    .stApp { background:{C_BG} !important; color:{C_TEXT} !important; font-family:'Inter',sans-serif; }
    .stApp > header { background:{C_BG} !important; border-bottom:1px solid {C_BORD} !important; }
    .block-container { padding:1.5rem 2rem 3rem !important; max-width:100% !important; }
    header, footer { visibility:hidden; }
    * { box-sizing:border-box; }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {
        background:{C_SIDEBAR} !important;
        border-right:1px solid {C_BORD} !important;
        min-width:260px !important;
    }
    section[data-testid="stSidebar"] .stSlider > div { color:{C_SUB} !important; }
    section[data-testid="stSidebar"] label { color:{C_SUB} !important; font-size:0.78rem !important; letter-spacing:0.08em; text-transform:uppercase; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p { color:{C_TEXT} !important; }

    /* ── SLIDER ── */
    .stSlider [data-baseweb="slider"] > div > div > div { background:{C_ACCENT} !important; }
    .stSlider [data-baseweb="slider"] > div > div:first-child { background:{C_BORD} !important; }

    /* ── TYPOGRAPHY ── */
    h1,h2,h3,h4,h5,h6 { color:{C_TEXT} !important; font-weight:700 !important; letter-spacing:-0.02em; }
    p { color:{C_SUB}; line-height:1.8; }

    /* ===== TABS (BIG NAVIGATION) ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px; background: {C_SIDEBAR}; padding: 12px; border-radius: 18px;
        border: 1px solid {C_BORD};
        margin-bottom: 35px; justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto; padding: 20px 36px; border-radius: 14px;
        font-size: 1.3rem; font-weight: 800; color: {C_SUB};
        transition: all 0.3s ease; white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"]:hover { color:{C_TEXT}; background:{C_HOVER}; }
    .stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] div {
        background:{C_ACCENT} !important;
        color:#FFFFFF !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding:0 !important; }

    /* ── METRIC CARD ── */
    .kpi-card {
        background:{C_SIDEBAR};
        border:1px solid {C_BORD};
        border-radius:14px;
        padding:22px 20px 18px;
        margin-bottom:12px;
        position:relative;
        overflow:hidden;
        transition:border-color 0.25s;
    }
    .kpi-card:hover { border-color:#3A3A5C; }
    .kpi-card .kpi-accent { position:absolute; top:0; left:0; right:0; height:2px; }
    .kpi-icon { font-size:1.3rem; margin-bottom:8px; opacity:0.75; }
    .kpi-val { font-size:1.6rem; font-weight:800; letter-spacing:-0.03em; margin:4px 0 3px; }
    .kpi-lbl { font-size:0.72rem; color:{C_SUB}; text-transform:uppercase; letter-spacing:0.12em; font-weight:600; }
    .kpi-delta { font-size:0.72rem; margin-top:4px; font-weight:600; }
    .kpi-delta.up { color:#4ADE80; }
    .kpi-delta.dn { color:#F87171; }

    /* ── SECTION HEADER ── */
    .section-title {
        font-size:0.72rem; color:{C_ACCENT}; font-weight:700;
        text-transform:uppercase; letter-spacing:0.14em;
        margin:2.5rem 0 1rem; display:flex; align-items:center; gap:8px;
    }
    .section-title::after { content:''; flex:1; height:1px; background:{C_BORD}; }

    /* ── INFO PILL ── */
    .info-pill {
        display:inline-flex; align-items:center; gap:6px;
        background:{C_CARD}; border:1px solid {C_BORD};
        border-radius:20px; padding:6px 14px;
        font-size:0.75rem; color:{C_SUB};
        margin-bottom:1.5rem;
    }
    .info-pill .dot { width:6px; height:6px; border-radius:50%; background:{C_ACCENT}; flex-shrink:0; }

    /* ── WINNER CARD ── */
    .winner-card {
        background:{C_CARD};
        border:1px solid {C_ACCENT};
        border-radius:12px; padding:18px 24px;
        display:flex; align-items:center; gap:16px;
        margin:16px 0;
    }
    .winner-badge {
        background:{C_ACCENT}; color:#fff;
        border-radius:8px; padding:10px 14px;
        font-size:1.4rem; line-height:1;
        flex-shrink:0;
    }
    .winner-title { font-size:1.1rem; font-weight:700; color:{C_TEXT}; margin-bottom:2px; }
    .winner-sub { font-size:0.78rem; color:{C_SUB}; }

    /* ── RMSE TABLE ── */
    .rmse-wrap { background:{C_SIDEBAR}; border:1px solid {C_BORD}; border-radius:12px; overflow:hidden; margin:12px 0; }
    .rmse-wrap table { width:100%; border-collapse:collapse; }
    .rmse-wrap th { background:{C_CARD}; color:{C_SUB}; padding:12px 16px; text-align:left; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:700; border-bottom:1px solid {C_BORD}; }
    .rmse-wrap td { padding:13px 16px; border-bottom:1px solid {C_CARD}; font-size:0.88rem; color:{C_SUB}; font-family:'JetBrains Mono',monospace; }
    .rmse-wrap tr:last-child td { border-bottom:none; }
    .rmse-wrap tr.best td { color:{C_TEXT}; background:{C_HOVER}; }
    .rmse-wrap tr.best td:first-child { color:{C_ACCENT}; font-weight:700; }
    .rmse-bar { height:4px; border-radius:2px; background:{C_ACCENT}; margin-top:4px; transition:width 0.6s ease; }

    /* ── ARCH CARD ── */
    .arch-card {
        background:{C_SIDEBAR}; border:1px solid {C_BORD};
        border-radius:14px; padding:22px 20px;
        height:100%; transition:border-color 0.2s;
    }
    .arch-card:hover { border-color:{C_ACCENT}; }
    .arch-title { font-size:1rem; font-weight:700; color:{C_TEXT}; margin-bottom:8px; }
    .arch-tag { display:inline-block; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; padding:3px 10px; border-radius:20px; margin-bottom:12px; }
    .arch-body { font-size:0.82rem; color:{C_SUB}; line-height:1.7; }
    .arch-prop { display:flex; align-items:flex-start; gap:8px; margin-top:8px; font-size:0.78rem; color:#808096; }
    .arch-prop .check { color:#4ADE80; flex-shrink:0; }
    .arch-prop .warn { color:#FBBF24; flex-shrink:0; }

    /* ── DATA TABLE ── */
    .stDataFrame { border-radius:12px; overflow:hidden; border:1px solid {C_BORD} !important; }
    .stDataFrame thead th { background:{C_CARD} !important; color:{C_SUB} !important; }
    .stDataFrame tbody tr { background:{C_SIDEBAR} !important; }

    /* ── EXPANDER ── */
    [data-testid="stExpander"] details summary {
        background:{C_SIDEBAR} !important;
        border:1px solid {C_BORD} !important;
        border-radius:10px !important;
        color:{C_TEXT} !important;
        font-weight:600 !important;
    }
    [data-testid="stExpander"] details summary p {
        color:{C_TEXT} !important;
        font-weight:600 !important;
    }
    [data-testid="stExpander"] details [data-testid="stExpanderDetails"] { background:{C_BG} !important; border:1px solid {C_BORD} !important; border-top:none !important; }

    /* ===== BUTTONS ===== */
    div[data-testid="stDownloadButton"] button, div[data-testid="stButton"] button, .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #2B6CB0, #3182CE) !important; color: #ffffff !important;
        border: none; border-radius: 12px; font-weight: 800; font-size: 1.15rem; padding: 0.9rem 2rem;
        transition: all 0.3s ease; width: 100%; letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(49,130,206,0.25);
    }
    div[data-testid="stDownloadButton"] button p, div[data-testid="stButton"] button p, .stButton>button p, .stDownloadButton>button p {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    div[data-testid="stDownloadButton"] button:hover, div[data-testid="stButton"] button:hover, .stButton>button:hover, .stDownloadButton>button:hover { background:#2B6CB0 !important; color:#ffffff !important; }
    div[data-testid="stDownloadButton"] button:hover p, div[data-testid="stButton"] button:hover p { color:#ffffff !important; }

    /* ── SPINNER ── */
    .stSpinner > div { border-top-color:{C_ACCENT} !important; }

    /* ── PROGRESS ── */
    .pipeline-step {
        display:flex; align-items:center; gap:14px;
        padding:14px 18px; background:{C_SIDEBAR};
        border:1px solid {C_BORD}; border-radius:10px; margin-bottom:8px;
    }
    .pipeline-step .step-num {
        width:28px; height:28px; border-radius:50%;
        background:{C_CARD}; border:1px solid {C_BORD};
        display:flex; align-items:center; justify-content:center;
        font-size:0.75rem; font-weight:700; color:{C_ACCENT};
        flex-shrink:0;
    }
    .pipeline-step .step-text { font-size:0.85rem; color:{C_SUB}; }
    .pipeline-step .step-label { font-size:0.72rem; color:{C_SUB}; margin-top:2px; }
    .pipeline-connector { width:1px; height:12px; background:{C_BORD}; margin:0 0 0 31px; }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width:5px; height:5px; }
    ::-webkit-scrollbar-track { background:{C_BG}; }
    ::-webkit-scrollbar-thumb { background:#2A2A3E; border-radius:3px; }

    /* ── HERO BANNER ── */
    .hero-bar {
        background:{C_SIDEBAR}; border:1px solid {C_BORD};
        border-radius:16px; padding:28px 32px;
        display:flex; align-items:center; justify-content:space-between;
        margin-bottom:2rem; flex-wrap:wrap; gap:16px;
    }
    .hero-title { font-size:1.8rem; font-weight:800; color:{C_TEXT}; letter-spacing:-0.04em; margin:0; }
    .hero-title span { color:{C_ACCENT}; }
    .hero-desc { font-size:0.82rem; color:{C_SUB}; margin-top:4px; }
    .hero-chips { display:flex; flex-wrap:wrap; gap:8px; }
    .chip {
        background:{C_CARD}; border:1px solid {C_BORD};
        border-radius:20px; padding:5px 13px;
        font-size:0.72rem; color:{C_SUB}; font-weight:600;
        letter-spacing:0.03em; white-space:nowrap;
    }
    .chip.active { background:#1A1A38; border-color:{C_ACCENT}; color:#8080F0; }
</style>
""".replace('{C_BG}', C_BG).replace('{C_CARD}', C_CARD).replace('{C_SIDEBAR}', C_SIDEBAR).replace('{C_TEXT}', C_TEXT).replace('{C_SUB}', C_SUB).replace('{C_BORD}', C_BORD).replace('{C_HOVER}', C_HOVER).replace('{C_ACCENT}', C_ACCENT), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# PLOTLY DARK THEME
# ──────────────────────────────────────────────────────────────────
CHART_COLORS = {
    'Retail':   '#5B5BD6',
    'US':       '#06B6D4',
    'IndiaCG':  '#34D399',
    'IndiaMfg': '#A78BFA',
    'ARIMA':    '#5B5BD6',
    'ETS':      '#06B6D4',
    'Random Forest': '#34D399',
}

def dark_fig(fig, title="", h=340):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#A0A0B4", size=13, family="Inter"), x=0, xanchor='left') if title else None,
        template=P_THEME,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=C_CARD,
        margin=dict(l=10, r=10, t=38 if title else 10, b=10),
        height=h,
        hovermode='x unified',
        hoverlabel=dict(bgcolor=C_CARD, bordercolor=C_BORD, font=dict(color=C_TEXT, size=12)),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(color='#6B6B80', size=11),
            linecolor=C_BORD,
        ),
        yaxis=dict(
            showgrid=True, gridcolor=C_BG, zeroline=False,
            tickfont=dict(color='#6B6B80', size=11),
            linecolor=C_BORD,
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(color='#A0A0B4', size=11),
            bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)',
        ),
        font=dict(family='Inter', color='#A0A0B4'),
    )
    return fig


def kpi(icon, label, value, accent="{C_ACCENT}", delta=None, delta_dir=None):
    delta_html = ""
    if delta:
        c = "#34D399" if delta_dir == "up" else "#F87171"
        arr = "▲" if delta_dir == "up" else "▼"
        delta_html = f'<div class="kpi-delta" style="color:{c};">{arr} {delta}</div>'

    st.markdown(f"""
    <div class="kpi-card" style="background:{C_CARD}; border:1px solid {C_BORD};">
        <div class="kpi-icon" style="color:{accent};">{icon}</div>
        <div class="kpi-val" style="color:{C_TEXT};">{value}</div>
        <div class="kpi-lbl" style="color:{C_SUB};">{label}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("<p style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;color:#5B5BD6;font-weight:700;margin-bottom:1rem'>⚙ Engine Controls</p>", unsafe_allow_html=True)

    split_ratio = st.slider("Train / Test Split", 60, 90, 80, 5,
                            format="%d%%", help="% of data used for training", disabled=True) / 100.0

    forecast_horizon = st.slider("Forecast Horizon", 3, 36, 12, 3,
                                 format="%d mo", help="Months to forecast ahead", disabled=True)

    st.markdown("---")
    st.markdown("<p style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;color:#5B5BD6;font-weight:700;margin-bottom:0.75rem'>Active Datasets</p>", unsafe_allow_html=True)
    show_retail = st.checkbox("Retail Sales", value=True)
    show_us     = st.checkbox("US Retail (Exog A)", value=True)
    show_cg     = st.checkbox("India Consumer Goods (Exog B)", value=True)
    show_mfg    = st.checkbox("India Manufacturing (Exog C)", value=True)

    st.markdown("---")
    st.markdown("<p style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;color:#5B5BD6;font-weight:700;margin-bottom:0.75rem'>Model Selection</p>", unsafe_allow_html=True)
    run_arima   = st.checkbox("ARIMA(1,1,1)", value=True)
    run_ets     = st.checkbox("ETS (Holt-Winters)", value=True)
    run_rf      = st.checkbox("Random Forest", value=True)
    rf_trees    = st.slider("RF Trees (n_estimators)", 20, 400, 100, 20,
                            help="Number of decision trees in the Random Forest ensemble",
                            disabled=not run_rf)

    st.markdown("---")
    st.markdown(f"""
    <div style='background:#131322;border:1px solid #1E1E2A;border-radius:10px;padding:14px 16px;'>
        <p style='font-size:0.72rem;color:#5B5BD6;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin:0 0 8px'>Validation Strategy</p>
        <p style='font-size:0.78rem;color:#6B6B80;margin:0;line-height:1.6'>
            Train on <b style='color:#A0A0B4'>{int(split_ratio*100)}%</b> → evaluate on 
            <b style='color:#A0A0B4'>{int((1-split_ratio)*100)}%</b> holdout →
            refit winner on full data → forecast.
        </p>
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df_r = pd.read_excel('retailSale.xlsx'); df_r.columns = ['date','value']
        df_r['date'] = pd.to_datetime(df_r['date']); df_r = df_r.sort_values('date').reset_index(drop=True)

        df_u = pd.read_csv('MRTSSM44000USS.csv'); df_u.columns = ['date','value']
        df_u['date'] = pd.to_datetime(df_u['date']); df_u = df_u.sort_values('date').reset_index(drop=True)

        df_c = pd.read_csv('INDPRMNCG02IXOBM.csv'); df_c.columns = ['date','value']
        df_c['date'] = pd.to_datetime(df_c['date']); df_c = df_c.sort_values('date').reset_index(drop=True)

        df_m = pd.read_csv('INDPRMNTO01IXOBM.csv'); df_m.columns = ['date','value']
        df_m['date'] = pd.to_datetime(df_m['date']); df_m = df_m.sort_values('date').reset_index(drop=True)
        return df_r, df_u, df_c, df_m
    except:
        return None, None, None, None

df_retail, df_us, df_ind_cg, df_ind_mfg = load_data()

ALL_DS = [
    {"df": df_retail, "title": "Retail Sales",           "color": "#5B5BD6", "tag": "Retail",   "show": show_retail},
    {"df": df_us,     "title": "US Retail Sales",         "color": "#06B6D4", "tag": "US",       "show": show_us},
    {"df": df_ind_cg, "title": "India Consumer Goods",    "color": "#34D399", "tag": "IndiaCG",  "show": show_cg},
    {"df": df_ind_mfg,"title": "India Manufacturing",     "color": "#A78BFA", "tag": "IndiaMfg", "show": show_mfg},
]
DS = [d for d in ALL_DS if d["show"] and d["df"] is not None]


# ──────────────────────────────────────────────────────────────────
# CORE ENGINE  (math-corrected)
# ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def engine(df_in, sr, hz, _run_arima=True, _run_ets=True, _run_rf=True, _rf_trees=100):
    ts = df_in.set_index('date')[['value']].resample('MS').mean().interpolate().reset_index()
    y = ts['value'].values; dates = ts['date'].values; n = len(y)
    # Guard: need enough data to leave a non-empty test set after the 24-obs min train floor
    trn = max(int(n * sr), 24)          # bumped min to 24 for seasonal ETS
    trn = min(trn, n - 1)               # FIX: never consume the whole series (avoids empty test set)
    tst = n - trn
    y_tr, y_te = y[:trn], y[trn:]
    d_tr, d_te = dates[:trn], dates[trn:]
    R = {}

    # ── ARIMA(1,1,1) ────────────────────────────────────────────
    if _run_arima:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            m = ARIMA(y_tr, order=(1,1,1)).fit()
            fc_test = np.asarray(m.forecast(steps=tst))
            rmse = float(np.sqrt(mean_squared_error(y_te, fc_test)))
            # Refit on full data
            mf = ARIMA(y, order=(1,1,1)).fit()
            fo = mf.get_forecast(steps=hz)
            fv  = np.asarray(fo.predicted_mean)
            ci  = np.asarray(fo.conf_int(alpha=0.05))   # naturally widens with horizon
            clo, chi = ci[:, 0], ci[:, 1]
            R['ARIMA'] = {'fc_test': fc_test, 'rmse': rmse, 'fc_fut': fv, 'ci_lo': clo, 'ci_hi': chi}
        except Exception as e:
            R['ARIMA'] = {'fc_test': np.full(tst, np.nan), 'rmse': np.inf,
                          'fc_fut': np.full(hz, np.nan), 'ci_lo': np.full(hz, np.nan), 'ci_hi': np.full(hz, np.nan)}

    # ── ETS (Holt-Winters) ──────────────────────────────────────
    if _run_ets:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            kw_tr   = dict(trend='add', seasonal='add', seasonal_periods=12) if len(y_tr) >= 24 else dict(trend='add', seasonal=None)
            kw_full = dict(trend='add', seasonal='add', seasonal_periods=12) if len(y)    >= 24 else dict(trend='add', seasonal=None)  # FIX: separate kw

            m = ExponentialSmoothing(y_tr, **kw_tr).fit()
            fc_test = np.asarray(m.forecast(steps=tst))
            rmse = float(np.sqrt(mean_squared_error(y_te, fc_test)))

            mf = ExponentialSmoothing(y, **kw_full).fit()
            fv = np.asarray(mf.forecast(steps=hz))

            # FIX: horizon-expanding CI via simulation (replaces flat ±1.96σ)
            try:
                sim = mf.simulate(nsimulations=hz, repetitions=500, error='add')
                clo = np.percentile(sim, 2.5,  axis=1)
                chi = np.percentile(sim, 97.5, axis=1)
            except Exception:
                # Fallback: linearly expanding band (still better than flat)
                res_std = float(np.std(y - np.asarray(mf.fittedvalues)))
                step_factors = np.sqrt(np.arange(1, hz + 1))
                clo = fv - 1.96 * res_std * step_factors
                chi = fv + 1.96 * res_std * step_factors

            R['ETS'] = {'fc_test': fc_test, 'rmse': rmse, 'fc_fut': fv, 'ci_lo': clo, 'ci_hi': chi}
        except Exception:
            R['ETS'] = {'fc_test': np.full(tst, np.nan), 'rmse': np.inf,
                        'fc_fut': np.full(hz, np.nan), 'ci_lo': np.full(hz, np.nan), 'ci_hi': np.full(hz, np.nan)}

    # ── Random Forest (lag-1, lag-2, month) ─────────────────────
    if _run_rf:
        try:
            df_f = ts.copy()
            df_f['lag1']  = df_f['value'].shift(1)
            df_f['lag2']  = df_f['value'].shift(2)
            df_f['month'] = df_f['date'].dt.month
            df_f = df_f.dropna().reset_index(drop=True)

            cut  = pd.Timestamp(d_tr[-1])
            mask = df_f['date'] <= cut
            tr_rf, te_rf = df_f[mask], df_f[~mask]
            cols = ['lag1', 'lag2', 'month']

            rf = RandomForestRegressor(n_estimators=_rf_trees, random_state=42).fit(tr_rf[cols], tr_rf['value'])
            fc_test = rf.predict(te_rf[cols])
            rmse = float(np.sqrt(mean_squared_error(te_rf['value'].values, fc_test)))

            # Refit on full data
            rf2 = RandomForestRegressor(n_estimators=_rf_trees, random_state=42).fit(df_f[cols], df_f['value'])
            l1, l2 = float(y[-1]), float(y[-2])
            ld = pd.Timestamp(dates[-1])
            fv_list = []
            for i in range(hz):
                nd = ld + pd.DateOffset(months=i + 1)
                # FIX: pass a named DataFrame row (avoids sklearn feature-name warnings / fragility)
                x_row = pd.DataFrame([[l1, l2, nd.month]], columns=cols)
                p  = float(rf2.predict(x_row)[0])
                fv_list.append(p); l2 = l1; l1 = p
            fv = np.array(fv_list)

            # FIX: horizon-expanding CI using in-sample residual + sqrt(step) scaling
            rs = float(np.std(df_f['value'].values - rf2.predict(df_f[cols])))
            step_factors = np.sqrt(np.arange(1, hz + 1))
            clo = fv - 1.96 * rs * step_factors
            chi = fv + 1.96 * rs * step_factors

            R['Random Forest'] = {
                'fc_test':  fc_test, 'rmse': rmse, 'fc_fut': fv, 'ci_lo': clo, 'ci_hi': chi,
                'te_dates': te_rf['date'].values, 'te_actual': te_rf['value'].values
            }
        except Exception:
            R['Random Forest'] = {'fc_test': np.full(tst, np.nan), 'rmse': np.inf,
                                  'fc_fut': np.full(hz, np.nan), 'ci_lo': np.full(hz, np.nan), 'ci_hi': np.full(hz, np.nan)}

    if not R:
        return None

    rmses = {k: v['rmse'] for k, v in R.items()}
    best  = min(rmses, key=rmses.get)
    fd    = [pd.Timestamp(dates[-1]) + pd.DateOffset(months=i + 1) for i in range(hz)]
    return {
        'ts': ts, 'trn': trn, 'd_tr': d_tr, 'd_te': d_te,
        'y_tr': y_tr, 'y_te': y_te,
        'R': R, 'rmses': rmses, 'best': best, 'best_rmse': rmses[best], 'fd': fd
    }


# ──────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────
if df_retail is None:
    st.error("⚠️  Data files not found. Place `retailSale.xlsx`, `MRTSSM44000USS.csv`, `INDPRMNCG02IXOBM.csv`, `INDPRMNTO01IXOBM.csv` in the app directory.")
    st.stop()

st.markdown(f"""
<div class="hero-bar">
    <div>
        <p class="hero-title">Nexus <span>Retail</span> Intelligence</p>
        <p class="hero-desc">ARIMA · ETS · Random Forest · Out-of-sample validation · 95% CI forecasting</p>
    </div>
    <div class="hero-chips">
        <span class="chip active">📊 {len(DS)} active datasets</span>
        <span class="chip">🧠 {sum([run_arima, run_ets, run_rf])} models</span>
        <span class="chip">🔭 {forecast_horizon}mo horizon</span>
        <span class="chip">✂️ {int(split_ratio*100)}/{int((1-split_ratio)*100)} split</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Overview",
    "🏎️  Showdown",
    "🚀  Forecast",
    "🧠  Architecture",
    "ℹ️  About",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab1:
    if not DS:
        st.warning("No datasets selected. Enable at least one from the sidebar.")
        st.stop()

    # ── Global KPIs ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Global Summary</div>', unsafe_allow_html=True)

    all_vals = np.concatenate([d["df"]["value"].dropna().values for d in DS])
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1: kpi("📦", "Total Records",      f"{sum(len(d['df']) for d in DS):,}",    "#5B5BD6")
    with col_k2: kpi("📈", "Global Peak",         f"{all_vals.max():,.0f}",                 "#34D399")
    with col_k3: kpi("📉", "Global Min",           f"{all_vals.min():,.0f}",                 "#F87171")
    with col_k4: kpi("⌀",  "Grand Average",       f"{all_vals.mean():,.0f}",                "#A78BFA")


    # ── Per-dataset detail rows ──────────────────────────────────
    st.markdown('<div class="section-title">Dataset Detail</div>', unsafe_allow_html=True)

    for d in DS:
        df  = d["df"]
        c   = d["color"]
        tag = d["tag"]

        with st.expander(f"**{d['title']}**  ·  {df['date'].min().year} – {df['date'].max().year}", expanded=False):
            k1,k2,k3,k4,k5 = st.columns(5)
            with k1: kpi("🗓️","Span",     f"{df['date'].min().year}–{df['date'].max().year}", c)
            with k2: kpi("📊","Points",   f"{len(df):,}", c)
            with k3: kpi("📈","Peak",     f"{df['value'].max():,.0f}", "#34D399")
            with k4: kpi("📉","Trough",   f"{df['value'].min():,.0f}", "#F87171")
            with k5: kpi("⌀", "Mean",     f"{df['value'].mean():,.0f}", "#A78BFA")

            # Candlestick-style OHLC per year
            df_y = df.copy(); df_y['year'] = df_y['date'].dt.year
            ohlc = df_y.groupby('year')['value'].agg(['first','max','min','last']).reset_index()
            ohlc.columns = ['year','open','high','low','close']

            fig_ohlc = go.Figure()

            si = int(len(df) * split_ratio)
            split_date = df.iloc[si]['date'] if si < len(df) else None

            # Area fill (proper hex -> rgba conversion)
            _h = c.lstrip('#')
            _fill = f'rgba({int(_h[0:2],16)},{int(_h[2:4],16)},{int(_h[4:6],16)},0.07)' if len(_h) == 6 else 'rgba(91,91,214,0.07)'
            fig_ohlc.add_trace(go.Scatter(
                x=df['date'], y=df['value'], name='Monthly',
                mode='lines', line=dict(color=c, width=1.5),
                fill='tozeroy',
                fillcolor=_fill,
            ))
            if split_date:
                fig_ohlc.add_vline(x=split_date, line_width=1, line_dash='dot', line_color='#F87171')
                fig_ohlc.add_annotation(
                    x=split_date, y=df['value'].max(),
                    text="train | test", showarrow=False,
                    font=dict(color='#F87171', size=10), yshift=8
                )
            dark_fig(fig_ohlc, h=240)
            st.plotly_chart(fig_ohlc, use_container_width=True)

            # Year-by-year bar
            fig_bar = go.Figure(go.Bar(
                x=ohlc['year'], y=ohlc['close'],
                marker_color=c, marker_opacity=0.8,
                name='Year-end value',
                hovertemplate='Year %{x}: %{y:,.0f}<extra></extra>'
            ))
            dark_fig(fig_bar, "Year-end values", h=200)
            st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — SHOWDOWN
# ══════════════════════════════════════════════════════════════════
with tab2:
    if not DS:
        st.warning("No datasets selected.")
        st.stop()

    st.markdown('<div class="section-title">Algorithm Arena · Out-of-Sample Battle</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-pill">
        <span class="dot"></span>
        Models trained on {int(split_ratio*100)}% of data — RMSE evaluated strictly on unseen {int((1-split_ratio)*100)}% holdout
    </div>""", unsafe_allow_html=True)

    for d in DS:
        df  = d["df"]
        c   = d["color"]

        with st.expander(f"**{d['title']}**", expanded=True):
            with st.spinner(f"Training models on {d['title']}…"):
                res = engine(df, split_ratio, forecast_horizon, run_arima, run_ets, run_rf, rf_trees)
            if res is None:
                st.warning("No models selected."); continue

            rmses = res['rmses']; best = res['best']
            # FIX: guard against all-models-failed (empty generator -> ValueError) and zero/falsy max
            _finite = [v for v in rmses.values() if v != np.inf]
            max_rmse = max(_finite) if _finite else 1.0
            if max_rmse == 0:
                max_rmse = 1.0

            # Winner card
            st.markdown(f"""
            <div class="winner-card">
                <div class="winner-badge">🏆</div>
                <div>
                    <div class="winner-title">{best} wins this dataset</div>
                    <div class="winner-sub">Lowest out-of-sample RMSE = {res['best_rmse']:,.2f}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # RMSE table with inline bar
            rows = ""
            for model, rmse in sorted(rmses.items(), key=lambda x: x[1]):
                cls    = "best" if model == best else ""
                badge  = " 🏆" if model == best else ""
                bar_w  = int((rmse / max_rmse) * 100) if rmse != np.inf else 100
                rmse_s = f"{rmse:,.2f}" if rmse != np.inf else "failed"
                bar_c  = C_ACCENT if model == best else C_BORD
                rows += (
                    f'<tr class="{cls}">'
                    f'<td>{model}{badge}</td>'
                    f'<td>{rmse_s}'
                    f'<div class="rmse-bar" style="width:{bar_w}%;background:{bar_c}"></div>'
                    f'</td></tr>'
                )

            st.markdown(
                '<div class="rmse-wrap"><table>'
                '<tr><th>Model</th><th>Out-of-Sample RMSE ↓</th></tr>'
                f'{rows}'
                '</table></div>',
                unsafe_allow_html=True,
            )

            # Per-model forecast plots
            model_names = [k for k in res['R']]
            cols = st.columns(len(model_names))
            for col, mn in zip(cols, model_names):
                with col:
                    mc  = CHART_COLORS.get(mn, '#A0A0B4')
                    mr  = res['R'][mn]
                    if mn == 'Random Forest' and 'te_dates' in mr:
                        td, ta = mr['te_dates'], mr['te_actual']
                    else:
                        td, ta = res['d_te'], res['y_te']

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=td, y=ta, name='Actual',
                                             line=dict(color='#4B4B60', width=1.5)))
                    fig.add_trace(go.Scatter(x=td, y=mr['fc_test'], name=mn,
                                             line=dict(color=mc, width=2, dash='dash')))
                    badge = " ✅" if mn == best else ""
                    dark_fig(fig, f"{mn}{badge}  RMSE {mr['rmse']:,.0f}", h=280)
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — FORECAST
# ══════════════════════════════════════════════════════════════════
with tab3:
    if not DS:
        st.warning("No datasets selected.")
        st.stop()

    st.markdown('<div class="section-title">Future Projections · Best Model · 95% Confidence Interval</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-pill">
        <span class="dot"></span>
        Winner refit on 100% data → {forecast_horizon}-month forecast · CI widens with horizon (horizon-aware uncertainty)
    </div>""", unsafe_allow_html=True)

    for d in DS:
        df = d["df"]; c = d["color"]
        with st.expander(f"**{d['title']}**", expanded=True):
            with st.spinner(f"Generating forecast for {d['title']}…"):
                res = engine(df, split_ratio, forecast_horizon, run_arima, run_ets, run_rf, rf_trees)
            if res is None:
                st.warning("No models selected."); continue

            best = res['best']; mr = res['R'][best]
            fd   = res['fd']
            fc   = mr['fc_fut']; lo = mr['ci_lo']; hi = mr['ci_hi']

            # KPIs
            k1,k2,k3,k4 = st.columns(4)
            with k1: kpi("🏆", "Best Model", best, c)
            with k2: 
                v = f"{res['rmses']['ARIMA']:,.1f}" if 'ARIMA' in res['rmses'] else "N/A"
                kpi("📐", "ARIMA RMSE", v, "#8080F0")
            with k3: 
                v = f"{res['rmses']['ETS']:,.1f}" if 'ETS' in res['rmses'] else "N/A"
                kpi("📊", "ETS RMSE", v, "#06B6D4")
            with k4: 
                v = f"{res['rmses']['Random Forest']:,.1f}" if 'Random Forest' in res['rmses'] else "N/A"
                kpi("🌲", "RF RMSE", v, "#34D399")

            # Main forecast chart
            hex_c = c.lstrip('#')
            r_int = int(hex_c[0:2],16); g_int = int(hex_c[2:4],16); b_int = int(hex_c[4:6],16)

            fig = go.Figure()
            # Historical (faint)
            fig.add_trace(go.Scatter(
                x=res['ts']['date'], y=res['ts']['value'], name='Historical',
                line=dict(color=C_HIST, width=1.5),
                hovertemplate='%{y:,.0f}<extra>Historical</extra>'
            ))
            # CI band
            fig.add_trace(go.Scatter(
                x=list(fd) + list(fd[::-1]),
                y=list(hi) + list(lo[::-1]),
                fill='toself',
                fillcolor=f'rgba({r_int},{g_int},{b_int},0.10)',
                line=dict(color='rgba(0,0,0,0)'),
                name='95% CI Band', showlegend=True,
                hoverinfo='skip'
            ))
            # Upper bound
            fig.add_trace(go.Scatter(
                x=fd, y=hi, name='Upper 95%',
                line=dict(color='#F87171', width=1, dash='dot'),
                hovertemplate='Upper: %{y:,.0f}<extra></extra>'
            ))
            # Lower bound
            fig.add_trace(go.Scatter(
                x=fd, y=lo, name='Lower 95%',
                line=dict(color='#34D399', width=1, dash='dot'),
                hovertemplate='Lower: %{y:,.0f}<extra></extra>'
            ))
            # Forecast line
            fig.add_trace(go.Scatter(
                x=fd, y=fc, name=f'Forecast ({best})',
                mode='lines+markers',
                line=dict(color=c, width=2.5),
                marker=dict(size=6, color='#0A0A0F', line=dict(width=2, color=c)),
                hovertemplate='Forecast: %{y:,.0f}<extra>' + best + '</extra>'
            ))
            dark_fig(fig, f"{forecast_horizon}-Month Forecast · {best}", h=400)
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table + download side-by-side
            col_t, col_d = st.columns([3, 1])
            with col_t:
                tdf = pd.DataFrame({
                    'Date':       [x.strftime('%b %Y') for x in fd],
                    'Forecast':   [f"{v:,.2f}" for v in fc],
                    'Lower 95%':  [f"{v:,.2f}" for v in lo],
                    'Upper 95%':  [f"{v:,.2f}" for v in hi],
                })
                html_table = tdf.to_html(index=False, border=0)
                st.markdown(f'<div class="rmse-wrap">{html_table}</div>', unsafe_allow_html=True)
            with col_d:
                st.markdown("<br>", unsafe_allow_html=True)
                exp_df = pd.DataFrame({'Date': fd, 'Forecast': fc, 'Lower_95': lo, 'Upper_95': hi})
                csv    = exp_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    f"⬇️ Download {d['tag']} CSV", data=csv,
                    file_name=f"forecast_{d['tag'].lower()}.csv",
                    mime='text/csv', key=f"dl_{d['tag']}"
                )


# ══════════════════════════════════════════════════════════════════
# TAB 4 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">System Architecture · Flow Diagram</div>', unsafe_allow_html=True)

    # ── Interactive Graphviz flow diagram with inline math ──────
    _tr_pct = int(split_ratio * 100); _te_pct = 100 - _tr_pct
    dot = f"""
    digraph pipeline {{
        rankdir=TB;
        bgcolor="transparent";
        node [shape=box style="rounded,filled" fontname="Inter" fontsize=11
              color="{C_BORD}" fontcolor="{C_TEXT}" fillcolor="{C_SIDEBAR}" margin="0.18,0.12"];
        edge [color="#5B5BD6" arrowsize=0.7 penwidth=1.3];

        ingest   [label="1 · Data Ingestion\nretailSale.xlsx + 3 CSVs"];
        prep     [label="2 · Preprocessing\ny_t = resample(MS) → interpolate" fillcolor="{C_CARD}"];
        split    [label="3 · Split\ntrain = first {_tr_pct}%  |  test = last {_te_pct}%" fillcolor="{C_CARD}"];

        arima    [label="4a · ARIMA(1,1,1)\nΔy_t = c + φΔy_(t-1) + θε_(t-1) + ε_t" color="#5B5BD6" fontcolor="#8080F0"];
        ets      [label="4b · ETS (Holt-Winters)\nŷ = ℓ_t + h·b_t + s_(t+h-m)" color="#06B6D4" fontcolor="#06B6D4"];
        rf       [label="4c · Random Forest\nlag1, lag2, month → {{rf_trees}} trees" color="#34D399" fontcolor="#34D399"];

        rmse     [label="5 · Out-of-Sample RMSE\nRMSE = √(Σ(y-ŷ)² / n_test)" fillcolor="{C_CARD}"];
        sel      [label="6 · Model Selection\nwinner = argmin RMSE" fillcolor="{C_CARD}" color="#5B5BD6"];
        refit    [label="7 · Refit on 100% data"];
        fcast    [label="8 · Forecast + 95% CI\n{forecast_horizon}-mo horizon, CI ∝ √step" fillcolor="{C_CARD}" color="#5B5BD6" fontcolor="#8080F0"];

        ingest -> prep -> split;
        split -> arima; split -> ets; split -> rf;
        arima -> rmse; ets -> rmse; rf -> rmse;
        rmse -> sel -> refit -> fcast;
    }}
    """.replace("{rf_trees}", str(rf_trees))
    st.graphviz_chart(dot, use_container_width=True)

    st.markdown('<div class="section-title">Mathematical Formulation</div>', unsafe_allow_html=True)

    with st.expander("**Preprocessing & Validation**", expanded=True):
        st.markdown("**Monthly resampling** (mean within each month-start bucket, linear gap fill):")
        st.latex(r"y_t = \frac{1}{|B_t|}\sum_{i \in B_t} x_i, \qquad B_t = \{i : \text{month}(x_i) = t\}")
        st.markdown("**Chronological split** — no shuffling, the test set is strictly the most recent observations:")
        st.latex(r"\text{train} = \{y_1,\dots,y_k\}, \quad \text{test} = \{y_{k+1},\dots,y_n\}, \quad k=\max(\lfloor n\cdot s\rfloor,\,24)")
        st.markdown("**Out-of-sample error** — evaluated only on the unseen test window:")
        st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n_{\text{test}}}\sum_{t=k+1}^{n}\left(y_t-\hat{y}_t\right)^2}")

    with st.expander("**ARIMA(1,1,1)** — AutoRegressive Integrated Moving Average"):
        st.markdown("First-difference to remove trend, then AR(1) + MA(1) on the differenced series:")
        st.latex(r"\Delta y_t = y_t - y_{t-1}")
        st.latex(r"\Delta y_t = c + \phi\,\Delta y_{t-1} + \theta\,\varepsilon_{t-1} + \varepsilon_t,\qquad \varepsilon_t \sim \mathcal{N}(0,\sigma^2)")
        st.markdown("Forecast confidence interval grows with horizon via the state-space variance:")
        st.latex(r"\hat{y}_{n+h} \pm 1.96\,\sqrt{\operatorname{Var}(\varepsilon_{n+h}\mid \mathcal{F}_n)}")

    with st.expander("**ETS** — Holt-Winters additive trend + seasonality"):
        st.markdown("Level, trend, and seasonal recursions (additive, period \(m=12\)):")
        st.latex(r"\ell_t = \alpha\,(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})")
        st.latex(r"b_t = \beta\,(\ell_t - \ell_{t-1}) + (1-\beta)\,b_{t-1}")
        st.latex(r"s_t = \gamma\,(y_t - \ell_{t-1} - b_{t-1}) + (1-\gamma)\,s_{t-m}")
        st.markdown("**h-step forecast:**")
        st.latex(r"\hat{y}_{t+h} = \ell_t + h\,b_t + s_{t+h-m\lceil h/m\rceil}")
        st.markdown("CI estimated empirically from **500 simulated sample paths** (percentile band), which widens naturally with horizon.")

    with st.expander("**Random Forest** — recursive multi-step regression"):
        st.markdown("Feature vector and ensemble prediction (average over \(T\) trees):")
        st.latex(r"\mathbf{x}_t = \big[\,y_{t-1},\; y_{t-2},\; \text{month}(t)\,\big]")
        st.latex(r"\hat{y}_t = \frac{1}{T}\sum_{j=1}^{T} h_j(\mathbf{x}_t)")
        st.markdown("**Recursive forecasting** — each prediction feeds the next step's lags:")
        st.latex(r"\hat{y}_{n+i} = f\big(\hat{y}_{n+i-1},\, \hat{y}_{n+i-2},\, \text{month}\big)")
        st.markdown("Uncertainty compounds with each recursive step (\(\sqrt{\text{step}}\) scaling of residual std \(\hat{\sigma}\)):")
        st.latex(r"\hat{y}_{n+i} \pm 1.96\,\hat{\sigma}\,\sqrt{i}")

    st.markdown('<div class="section-title">Model Architectures</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="arch-card">
            <span class="arch-tag" style="background:#1A1A38;color:#8080F0">ARIMA(1,1,1)</span>
            <div class="arch-title">AutoRegressive Integrated Moving Average</div>
            <div class="arch-body">Captures linear trends and autocorrelation in first-differenced data.</div>
            <div class="arch-prop"><span class="check">✓</span> Native horizon-expanding CI from state-space representation</div>
            <div class="arch-prop"><span class="check">✓</span> Best for stable, trend-driven linear series</div>
            <div class="arch-prop"><span class="warn">~</span> Assumes linear dynamics; misses nonlinearity</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="arch-card">
            <span class="arch-tag" style="background:#0A2530;color:#06B6D4">ETS (Holt-Winters)</span>
            <div class="arch-title">Error · Trend · Seasonality</div>
            <div class="arch-body">Additive trend + additive 12-month seasonality. Exponentially weights recent observations.</div>
            <div class="arch-prop"><span class="check">✓</span> CI via simulation (500 paths) — widens with horizon</div>
            <div class="arch-prop"><span class="check">✓</span> Excels when seasonal patterns dominate</div>
            <div class="arch-prop"><span class="warn">~</span> Additive only; no multiplicative seasonality here</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="arch-card">
            <span class="arch-tag" style="background:#0A2518;color:#34D399">Random Forest</span>
            <div class="arch-title">Ensemble of 100 Decision Trees</div>
            <div class="arch-body">Features: lag-1, lag-2, calendar month. Captures nonlinear patterns via recursive multi-step prediction.</div>
            <div class="arch-prop"><span class="check">✓</span> CI expands as √(step) · residual std</div>
            <div class="arch-prop"><span class="check">✓</span> Captures nonlinear seasonality</div>
            <div class="arch-prop"><span class="warn">~</span> Cannot extrapolate beyond training range</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Math Corrections Applied</div>', unsafe_allow_html=True)

    fixes = [
        ("ETS CI", "Replaced flat ±1.96σ with simulation-based CI (500 paths) that widens with forecast horizon.", "#34D399"),
        ("RF CI",  "Replaced constant residual band with √(step)-scaled uncertainty — uncertainty compounds at each recursive step.", "#34D399"),
        ("ETS kw", "Separated kw_tr (sized on train) from kw_full (sized on full series) to prevent seasonal information loss.", "#34D399"),
        ("Min train", "Raised minimum training obs from 12 → 24 to guarantee at least 2 full seasonal cycles before fitting ETS.", "#34D399"),
    ]
    for name, desc, col in fixes:
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="step-num" style="background:#0A2518;border-color:#34D399;color:#34D399">✓</div>
            <div>
                <div class="step-text" style="color:#34D399">{name}</div>
                <div class="step-label">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════
with tab5:
        st.markdown(f"""
<div style='text-align:center; padding:60px 30px; background:{C_BG}; border:1px solid {C_BORD}; border-radius:16px; margin:2rem auto 0;'>
    <h1 style='color:{C_ACCENT}; margin-bottom:15px; font-size: 2.2rem;'>Study on Forecast Model for Predicting Retail Sales</h1>
    <p style='font-size:1.15rem; color:{C_SUB}; font-style:italic; max-width:800px; margin:0 auto 40px auto;'>
        Submitted in partial fulfillment of the requirements for the degree of<br>
        M.Tech in Industrial Engineering and Management
    </p>
    <div style="display:flex; justify-content:center; gap: 40px; margin-bottom: 40px; flex-wrap: wrap;">
        <div style="background: {C_CARD}; padding: 25px 40px; border-radius: 16px; border: 1px solid {C_BORD}; border-top: 4px solid {C_ACCENT};">
            <h4 style="color:{C_SUB}; margin-bottom:10px; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px;">Presented By</h4>
            <h2 style="color:{C_TEXT}; margin:0 0 8px 0; font-size: 1.6rem;">Sk. Najib Hossain</h2>
            <p style="margin:0; font-size:1.1rem; color:{C_SUB};">M.Tech, 3<sup>rd</sup> Semester (2<sup>nd</sup> Year)</p>
            <p style="margin:0; font-size:1rem; color:{C_SUB};">(University Roll No. 10013224004)</p>
        </div>
        <div style="background: {C_CARD}; padding: 25px 40px; border-radius: 16px; border: 1px solid {C_BORD}; border-top: 4px solid #38A169;">
            <h4 style="color:{C_SUB}; margin-bottom:10px; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px;">Under the Supervision of</h4>
            <h2 style="color:{C_TEXT}; margin:0 0 8px 0; font-size: 1.6rem;">Dr. Sourav Das</h2>
            <p style="margin:0; font-size:1.1rem; color:{C_SUB}; max-width:250px;">Department of Industrial Engineering and Management</p>
        </div>
    </div>
    <div style="padding-top: 30px; border-top: 1px solid {C_BORD};">
        <h3 style="color:{C_TEXT}; font-size:1.2rem; margin:0;">Maulana Abul Kalam Azad University of Technology, West Bengal</h3>
    </div>
</div>
""", unsafe_allow_html=True)
