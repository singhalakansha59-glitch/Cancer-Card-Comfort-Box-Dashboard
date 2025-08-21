import re
from pathlib import Path
from typing import Optional, Iterable, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# Page setup & styling
# =========================
st.set_page_config(page_title="Comfort Box Dashboard", page_icon="📦", layout="wide")

PRIMARY = "#1F77B4"   # deep blue
ACCENT1 = "#E4572E"   # warm red
ACCENT2 = "#2CA02C"   # green
ACCENT3 = "#F0A202"   # amber
NEUTRAL = "#4D4D4D"   # grey for grid/axes

st.markdown(
    """
    <style>
      .stApp {
        background: #FFFFFF !important;
        color: #000000 !important;
      }
      h1, h2, h3, h4, h5, h6, .metric-label, .metric-value {
        color: #000000 !important;
        font-weight: 700 !important;
      }
      .block-container {
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
      }
      .card {
        background: #FFFFFF;
        border: 1px solid #EAEAEA;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,.05);
        margin-bottom: 14px;
      }
      .section-title {
        font-weight: 800; font-size: 1.2rem; margin-bottom: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Comfort Box Dashboard")

# =========================
# Helpers: robust loader
# =========================
def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.replace(" ", "_")

ALIASES = {
    # core dates seen in your CSV
    "requested_date": {"requested_date","request_date","date_requested","comfort_box_requested_date"},
    "sent_date": {"sent_date","date_sent","dispatch_date","comfort_box_sent_date"},
    "received_date": {"received_date","date_received","delivery_date","delivered_date","comfort_box_received_date"},
    "created_date": {"journal_entry_created_date"},
    "postal_code": {"postal_code","postcode","postalcode","post_code"},
    # survey fields
    "satisfaction": {
        "1_on_a_scale_of_1_10_how_would_you_rate_your_overall_satisfaction_with_the_comfort_box_1_not_satisfied_to_10_extremely_satisfied"
    },
    "valuable_item": {"2_which_item_from_the_comfort_box_did_you_find_most_valuable"},
    "met_expectations": {
        "3_did_the_comfort_box_meet_your_expectations_in_terms_of_providing_practical_assistance_and_comfort_during_your_cancer_journey"
    },
    "would_recommend": {"6_would_you_recommend_the_cancer_card_comfort_box_to_someone_else_going_through_cancer_treatment"},
    "comments": {"7_any_additional_comments_or_suggestions","note"},
}

def resolve(df_cols: Iterable[str], cand_set: set) -> Optional[str]:
    norm_cols = {_norm(c): c for c in df_cols}
    for c in cand_set:
        if c in norm_cols:
            return norm_cols[c]
    return None

@st.cache_data(show_spinner=False)
def load_data(uploaded) -> pd.DataFrame:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        # try common paths; change as needed
        for p in (
            Path("./Comfort box Data.csv"),
            Path("./data/Comfort box Data.csv"),
            Path("/mnt/data/Comfort box Data.csv"),
        ):
            if p.exists():
                df = pd.read_csv(p)
                break
        else:
            st.error("Upload **Comfort box Data.csv** or place it in ./, ./data/, or /mnt/data/.")
            st.stop()

    cols = list(df.columns)

    # map resolved names to canonical keys
    mapping: Dict[str, str] = {}
    for key, cand in ALIASES.items():
        found = resolve(cols, cand)
        if found:
            mapping[key] = found

    # rename to canonical column names used below
    rename = {}
    if "requested_date" in mapping: rename[mapping["requested_date"]] = "RequestedDate"
    if "sent_date"      in mapping: rename[mapping["sent_date"]]      = "SentDate"
    if "received_date"  in mapping: rename[mapping["received_date"]]  = "ReceivedDate"
    if "created_date"   in mapping: rename[mapping["created_date"]]   = "CreatedDate"
    if "postal_code"    in mapping: rename[mapping["postal_code"]]    = "PostalCode"
    if "satisfaction"   in mapping: rename[mapping["satisfaction"]]   = "Satisfaction"
    if "valuable_item"  in mapping: rename[mapping["valuable_item"]]  = "ValuableItem"
    if "met_expectations" in mapping: rename[mapping["met_expectations"]] = "MetExpectations"
    if "would_recommend" in mapping: rename[mapping["would_recommend"]] = "WouldRecommend"
    if "comments"       in mapping: rename[mapping["comments"]]       = "Comments"

    df = df.rename(columns=rename).copy()

    # parse dates if present
    for c in ["RequestedDate","SentDate","ReceivedDate","CreatedDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # derive intervals if possible
    if {"RequestedDate","SentDate"}.issubset(df.columns):
        df["Request_to_Send"] = (df["SentDate"] - df["RequestedDate"]).dt.days
    if {"SentDate","ReceivedDate"}.issubset(df.columns):
        df["Send_to_Receive"] = (df["ReceivedDate"] - df["SentDate"]).dt.days
    if {"RequestedDate","ReceivedDate"}.issubset(df.columns):
        df["Request_to_Receive"] = (df["ReceivedDate"] - df["RequestedDate"]).dt.days

    # satisfaction numeric if present
    if "Satisfaction" in df.columns:
        df["Satisfaction"] = pd.to_numeric(df["Satisfaction"], errors="coerce")

    # pick a default date for grouping
    for candidate in ["RequestedDate","SentDate","ReceivedDate","CreatedDate"]:
        if candidate in df.columns:
            df["GroupDate"] = df[candidate]
            break

    return df

# =========================
# Sidebar: upload & filters
# =========================
uploaded = st.sidebar.file_uploader("Upload Comfort box Data.csv", type=["csv"])
df = load_data(uploaded)

st.sidebar.subheader("Filters")
min_date = df["GroupDate"].min() if "GroupDate" in df.columns else None
max_date = df["GroupDate"].max() if "GroupDate" in df.columns else None
if min_date is not None and max_date is not None:
    start, end = st.sidebar.date_input(
        "Date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    mask = (df["GroupDate"].dt.date >= start) & (df["GroupDate"].dt.date <= end)
    fdf = df.loc[mask].copy()
else:
    fdf = df.copy()

# =========================
# KPI row (non-overlapping)
# =========================
k1, k2, k3, k4 = st.columns(4)
with k1:
    with st.container(border=True):
        st.metric(label="Total Requests", value=f"{len(fdf):,}")
with k2:
    with st.container(border=True):
        med_r2s = np.nanmedian(fdf["Request_to_Send"]) if "Request_to_Send" in fdf.columns else np.nan
        st.metric(label="Median Days: Request→Send", value="—" if np.isnan(med_r2s) else f"{med_r2s:.0f} d")
with k3:
    with st.container(border=True):
        med_s2r = np.nanmedian(fdf["Send_to_Receive"]) if "Send_to_Receive" in fdf.columns else np.nan
        st.metric(label="Median Days: Send→Receive", value="—" if np.isnan(med_s2r) else f"{med_s2r:.0f} d")
with k4:
    with st.container(border=True):
        avg_sat = np.nanmean(fdf["Satisfaction"]) if "Satisfaction" in fdf.columns else np.nan
        st.metric(label="Average Satisfaction (1–10)", value="—" if np.isnan(avg_sat) else f"{avg_sat:.1f}")

# =========================
# Gauges (clear, side-by-side)
# =========================
g1, g2, g3 = st.columns(3)

def gauge(title, value, minv=0, maxv=100, color=PRIMARY):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0 if value is None or np.isnan(value) else float(value),
        number={'font': {'color': '#000000', 'size': 20}},
        gauge={
            'axis': {'range': [minv, maxv], 'tickcolor': NEUTRAL, 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#EAEAEA",
        },
        title={'text': f"<b>{title}</b>", 'font': {'color': '#000000', 'size': 16}}
    ))
    fig.update_layout(height=240, margin=dict(t=40, b=10, l=10, r=10), paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

with g1:
    if "Satisfaction" in fdf.columns:
        gauge("Satisfaction (Avg)", np.nanmean(fdf["Satisfaction"]), minv=0, maxv=10, color=ACCENT2)
with g2:
    if "Request_to_Send" in fdf.columns and fdf["Request_to_Send"].notna().any():
        on_time_pct = np.mean(fdf["Request_to_Send"] <= 7) * 100
        gauge("Sent within 7 days (%)", on_time_pct, minv=0, maxv=100, color=PRIMARY)
with g3:
    if "WouldRecommend" in fdf.columns:
        rec_pct = np.mean(fdf["WouldRecommend"].astype(str).str.lower().isin(["yes","y","true","1"])) * 100
        gauge("Would Recommend (%)", rec_pct, minv=0, maxv=100, color=ACCENT1)

# =========================
# Trends (no overlaps)
# =========================
st.markdown('<div class="section-title">Trends</div>', unsafe_allow_html=True)
t1, t2 = st.columns(2)

with t1:
    if "GroupDate" in fdf.columns:
        by_month = (
            fdf.set_index("GroupDate")
               .resample("M")
               .size()
               .rename("Requests")
               .reset_index()
        )
        fig = px.line(by_month, x="GroupDate", y="Requests", markers=True, template="plotly_white")
        fig.update_traces(line=dict(color=PRIMARY, width=3), marker=dict(size=7))
        fig.update_layout(
            title="<b>Requests per Month</b>",
            height=360, margin=dict(t=60, b=10, l=10, r=10),
            font=dict(color="#000000"),
            xaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Month"),
            yaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Count"),
            paper_bgcolor="white", plot_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    else:
        st.info("No date column available for monthly trend.")

with t2:
    if "ValuableItem" in fdf.columns:
        s = (
            fdf["ValuableItem"].astype(str).str.strip()
            .replace({"": np.nan, "nan": np.nan}).dropna()
        )
        if s.str.contains(r"[;,]").any():
            s = s.str.split(r"[;,]").explode().str.strip()
            s = s.replace({"": np.nan}).dropna()
        counts = s.value_counts().reset_index()
        counts.columns = ["Item", "Count"]
        counts = counts.sort_values("Count", ascending=True)
        fig = px.bar(
            counts, x="Count", y="Item", orientation="h",
            template="plotly_white", color_discrete_sequence=[PRIMARY]
        )
        fig.update_layout(
            title="<b>Most Valuable Items</b>",
            height=360, margin=dict(t=60, b=10, l=10, r=10),
            font=dict(color="#000000"),
            xaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Responses"),
            yaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Item"),
            paper_bgcolor="white", plot_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    else:
        st.info("No ‘most valuable item’ responses found.")

# =========================
# SLA / intervals
# =========================
st.markdown('<div class="section-title">Fulfilment Intervals</div>', unsafe_allow_html=True)
i1, i2 = st.columns(2)

def interval_hist(col, title, color):
    data = fdf[col].dropna() if col in fdf.columns else pd.Series([], dtype=float)
    if data.empty:
        st.info(f"No data for {title}.")
        return
    fig = px.histogram(data, nbins=20, template="plotly_white", color_discrete_sequence=[color])
    fig.update_layout(
        title=f"<b>{title}</b>",
        height=360, margin=dict(t=60, b=10, l=10, r=10),
        font=dict(color="#000000"),
        xaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Days"),
        yaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Count"),
        paper_bgcolor="white", plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

with i1:
    interval_hist("Request_to_Send", "Request → Send (days)", ACCENT3)
with i2:
    interval_hist("Send_to_Receive", "Send → Receive (days)", ACCENT2)

# =========================
# Expectations / Recommendation
# =========================
st.markdown('<div class="section-title">Expectations & Recommendation</div>', unsafe_allow_html=True)
e1, e2 = st.columns(2)

with e1:
    if "MetExpectations" in fdf.columns:
        s = fdf["MetExpectations"].astype(str).str.strip().str.title()
        s = s.replace({"": np.nan, "Nan": np.nan}).dropna()
        fig = px.histogram(s, template="plotly_white", color_discrete_sequence=[ACCENT1])
        fig.update_layout(
            title="<b>Did the Comfort Box meet expectations?</b>",
            height=360, margin=dict(t=60, b=10, l=10, r=10),
            font=dict(color="#000000"),
            xaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Response"),
            yaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Count"),
            paper_bgcolor="white", plot_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    else:
        st.info("No ‘met expectations’ responses found.")

with e2:
    if "WouldRecommend" in fdf.columns:
        s = fdf["WouldRecommend"].astype(str).str.strip().str.title()
        s = s.replace({"": np.nan, "Nan": np.nan}).dropna()
        fig = px.histogram(s, template="plotly_white", color_discrete_sequence=[PRIMARY])
        fig.update_layout(
            title="<b>Would you recommend the Comfort Box?</b>",
            height=360, margin=dict(t=60, b=10, l=10, r=10),
            font=dict(color="#000000"),
            xaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Response"),
            yaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Count"),
            paper_bgcolor="white", plot_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    else:
        st.info("No recommendation responses found.")

# =========================
# Comments Table (optional)
# =========================
with st.expander("View Comments"):
    if "Comments" in fdf.columns:
        st.dataframe(fdf[["Comments"]].dropna().reset_index(drop=True), use_container_width=True)
    else:
        st.caption("No comments column detected.")
