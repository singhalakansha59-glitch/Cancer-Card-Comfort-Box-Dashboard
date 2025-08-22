

from __future__ import annotations
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Comfort Box Dashboard", page_icon="📦", layout="wide")
st.markdown("""
<style>
  :root, .stApp { background:#f7fafc; color:#000; }
  .block-container{ padding-top:.6rem; padding-bottom:.8rem; }
  .title{ text-align:center; font-weight:900; font-size:1.4rem; letter-spacing:.02em; margin:.2rem 0 .8rem; color:#000;}
  .card{ background:#fff; border:1px solid #e5e7eb; border-radius:14px;
         box-shadow:0 12px 28px rgba(2,8,20,.10); padding:14px 16px; margin:12px 0; }
  h3{ margin:0 0 8px 2px; color:#000; font-size:1.0rem; font-weight:800;}
  .stPlotlyChart > div, .js-plotly-plot{
      border-radius:10px; background:#fff;
      box-shadow:0 10px 24px rgba(2,8,20,.12); padding:6px;
  }
</style>
""", unsafe_allow_html=True)
st.markdown("<div class='title'>COMFORT BOX ANALYSIS</div>", unsafe_allow_html=True)

COL_CREATED  = "Journal Entry Created Date"
COL_AWARE    = "How did you find out about Cancer Card"
COL_NOTE     = "Note"
COL_Q1_SAT   = "1. On a scale of 1-10, how would you rate your overall satisfaction with the Comfort Box? 1 (Not Satisfied) to 10 (Extremely Satisfied)"
COL_Q2_ITEM  = "2. Which item from the Comfort Box did you find most valuable?"
COL_Q3_EXPECT= "3. Did the Comfort Box meet your expectations in terms of providing practical assistance and comfort during your cancer journey?"
COL_Q4_CHANGE= "4. Is there any specific item you would like to see added or changed in the Comfort Box?"
COL_Q5_EMO   = "5. How did the Comfort Box impact your emotional well-being during your cancer treatment?"
COL_Q6_RECO  = "6. Would you recommend the Cancer Card Comfort Box to someone else going through cancer treatment?"
COL_Q7_TEXT  = "7. Any additional comments or suggestions?"
COL_REQ      = "Comfort Box Requested Date"
COL_SENT     = "Comfort Box Sent Date"
COL_POSTCODE = "Postal Code"


GRID_COLS = 2
H = 420


C1 = ["#2563eb"]; C2 = ["#06b6d4"]; C3 = ["#0ea5e9"]
C4 = ["#f59e0b"]; C5 = ["#ef4444"]; C6 = ["#22c55e"]
C8 = ["#a855f7"]; C9 = ["#84cc16"]
MULTI  = px.colors.qualitative.Set3
MULTI2 = px.colors.qualitative.Bold

def style_fig(fig, *, showgrid=True, height=H, title_text=None,
              x_title=None, y_title=None, legend=False):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=42, b=10),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(color="#000", size=13),
        showlegend=legend,
        uniformtext_minsize=10, uniformtext_mode="hide",
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zeroline=False, showgrid=showgrid,
                     title_font=dict(color="#000", size=13), tickfont=dict(color="#000", size=12),
                     title_text=x_title if x_title is not None else fig.layout.xaxis.title.text)
    fig.update_yaxes(gridcolor="#e5e7eb", zeroline=False, showgrid=showgrid,
                     title_font=dict(color="#000", size=13), tickfont=dict(color="#000", size=12),
                     title_text=y_title if y_title is not None else fig.layout.yaxis.title.text)
    if title_text is not None:
        fig.update_layout(title=dict(text=title_text, font=dict(size=16, color="#000")))
    return fig

def put_card(col, title, render_fn):
    with col.container():
        st.markdown(f"<div class='card'><h3>{title}</h3>", unsafe_allow_html=True)
        render_fn()
        st.markdown("</div>", unsafe_allow_html=True)

CANCER_PATTERNS = [
    "breast cancer","lung cancer","prostate cancer","colorectal cancer","bowel cancer",
    "ovarian cancer","cervical cancer","endometrial cancer","uterine cancer","pancreatic cancer",
    "brain cancer","kidney cancer","renal cancer","bladder cancer","testicular cancer",
    "thyroid cancer","skin cancer","melanoma","leukaemia","leukemia","lymphoma",
    "multiple myeloma","myeloma","sarcoma","head and neck cancer","oral cancer","throat cancer",
    "stomach cancer","gastric cancer","oesophageal cancer","esophageal cancer","liver cancer",
    "gallbladder cancer"
]
CANCER_REGEXES = [re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE) for p in CANCER_PATTERNS]
STAGE_MAP = {
    r"\bstage\s*I\b|\bstage\s*1\b": "Stage I",
    r"\bstage\s*II\b|\bstage\s*2\b": "Stage II",
    r"\bstage\s*III\b|\bstage\s*3\b": "Stage III",
    r"\bstage\s*IV\b|\bstage\s*4\b": "Stage IV",
    r"\bmetastatic\b|\bmetastasis\b": "Metastatic"
}
STAGE_REGEXES: List[Tuple[re.Pattern,str]] = [(re.compile(p, re.IGNORECASE), v) for p, v in STAGE_MAP.items()]

def extract_cancers(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip(): return []
    lower = text.lower()
    found = []
    for rx, raw in zip(CANCER_REGEXES, CANCER_PATTERNS):
        if rx.search(lower): found.append(raw.title())
    seen, uniq = set(), []
    for x in found:
        if x not in seen: uniq.append(x); seen.add(x)
    return uniq

def extract_stage(text: str) -> str|None:
    if not isinstance(text, str) or not text.strip(): return None
    for rx, label in STAGE_REGEXES:
        if rx.search(text): return label
    return None

def is_yes_or_partial(v: str) -> bool:
    s = str(v).strip().lower()
    return s.startswith("yes") or s.startswith("partial")

NO_CHANGE_PATTERNS = [
    r"^no\b", r"\bnope\b", r"\bnothing\b", r"not that i can think of",
    r"none", r"no, everything is (fine|great|good)", r"n/?a\b", r"nil\b",
    r"can't think of", r"cannot think of", r"no suggestions?", r"no suggestion"
]
NO_CHANGE_RX = re.compile("|".join(NO_CHANGE_PATTERNS), re.IGNORECASE)
def is_no_change(text: str) -> bool:
    return isinstance(text, str) and bool(NO_CHANGE_RX.search(text.strip()))

def map_concise_suggestion(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "Other (misc)"
    s = text.lower()
    if "beanie" in s and "glove" in s: return "Beanie & gloves"
    if "leaflet" in s and ("hat" in s or "hats" in s or "eyelash" in s): return "Leaflet: hats & eyelashes"
    if "bigger blanket" in s or ("blanket" in s and ("bigger" in s or "larger" in s)): return "Bigger blanket"
    if ("warm" in s and "sock" in s) or "warm socks" in s: return "Warm socks"
    if "hot water bottle" in s: return "Hot water bottle"
    if "smaller slipper" in s or "smaller slippers" in s or ("slipper" in s and "small" in s): return "Smaller slippers"
    if "aromatherapy" in s or "essential oil" in s: return "Aromatherapy products"
    if "compact towel" in s or ("towel" in s and ("compact" in s or "small" in s or "travel" in s)): return "Compact towel"
    if ("stress" in s and "ball" in s) or "stress relief ball" in s: return "Stress relief ball"
    if any(k in s for k in ["warming gel","heat pad","heat pack","heated pad","hot pack","gel pack"]): return "Heat pack / warming gel"
    if "boiled sweet" in s or "boiled sweets" in s: return "Wee bag of boiled sweets"
    if any(k in s for k in ["tea","hot chocolate","hot choc"]): return "Tea/Hot chocolate selection"
    if "scarf" in s and any(k in s for k in ["colour","color","choice"]): return "Scarf – color choice"
    if "scarf" in s: return "Scarf"
    if "sock" in s: return "Wool socks"
    if any(k in s for k in ["hat","beanie","headgear","head gear","headwear","head wear"]): return "Headgear (hat/beanie)"
    if "glove" in s: return "Gloves"
    if "lip balm" in s or "hand cream" in s: return "Lip balm & hand cream"
    if any(k in s for k in ["puzzle","colouring","coloring"]): return "Puzzle/Adult colouring book"
    if "recipe" in s or "soup" in s: return "Recipe ideas for soup"
    if "leaflet" in s or "information" in s or "info leaflet" in s: return "Information leaflet"
    if any(k in s for k in ["phone charger","phone charge","charger","power bank"]): return "Phone charger / Power bank"
    if "tissue" in s: return "Tissues pack"
    if "thermometer" in s: return "Digital thermometer"
    if "blanket" in s: return "Wool blanket"
    if "travel mug" in s or re.search(r"\bmug\b", s): return "Travel mug"
    if "notebook" in s or "note book" in s or re.search(r"\bpen\b", s): return "Notebook & pen"
    return "Other (misc)"

# --- Awareness mapping (explicit sources first) ---
AWARE_ORDER = [
    "Beatson Cancer Wellbeing Centre, Glasgow",
    "Through Sharyn Black",
    "From Work",
    "Breast Cancer Now forum",
    "Macmillan Cancer Unit",
    "Maggie’s Centre",
    "Leaflet",
    "Western General Hospital Edinburgh",
    "NHCISC",
    "Through Nurse",
    "North Highland Cancer Centre",
    "Online",
    "GP Practice",
    "Hospital",
    "Friend",
    "Family",
    "Other",
]
def categorize_awareness(text: str) -> str:
    if not isinstance(text, str) or not text.strip(): return "Other"
    s = text.strip().lower()
    if "beatson" in s and ("wellbeing" in s or "centre" in s or "center" in s or "glasgow" in s):
        return "Beatson Cancer Wellbeing Centre, Glasgow"
    if "sharyn black" in s: return "Through Sharyn Black"
    if any(k in s for k in ["from work","at work","employer","colleague at work","work"]): return "From Work"
    if "breast cancer now" in s: return "Breast Cancer Now forum"
    if any(k in s for k in ["macmillan","mcmillan"]): return "Macmillan Cancer Unit"
    if any(k in s for k in ["maggie’s","maggies","maggie's"]): return "Maggie’s Centre"
    if "leaflet" in s or "leaflets" in s: return "Leaflet"
    if "western general" in s or ("wgh" in s and "edinburgh" in s): return "Western General Hospital Edinburgh"
    if "nhcisc" in s: return "NHCISC"
    if "nurse" in s: return "Through Nurse"
    if "north highland cancer centre" in s or "north highland cancer center" in s: return "North Highland Cancer Centre"
    if any(k in s for k in ["online","website","google","internet","facebook","instagram","twitter","x.com","tiktok","linkedin","social"]): return "Online"
    if any(k in s for k in ["gp","general practitioner","doctor","dr ","dr.","practice","surgery","primary care"]): return "GP Practice"
    if any(k in s for k in ["hospital","nhs","clinic","ward","oncolog","consultant"]): return "Hospital"
    if any(k in s for k in ["friend","colleague","coworker","mate"]): return "Friend"
    if any(k in s for k in ["family","mother","father","mum","dad","sister","brother","husband","wife","partner","relative"]): return "Family"
    return "Other"

# ---------------- Load & Prepare ----------------
@st.cache_data
def load_data(path="Comfort Box Data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Parse dates
    for c in [COL_CREATED, COL_REQ, COL_SENT]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # Clean text / numerics
    for c in [COL_NOTE, COL_AWARE, COL_Q2_ITEM, COL_Q3_EXPECT, COL_Q4_CHANGE, COL_Q5_EMO, COL_Q6_RECO, COL_Q7_TEXT]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace("nan","").str.strip()
    if COL_Q1_SAT in df.columns:
        df[COL_Q1_SAT] = pd.to_numeric(df[COL_Q1_SAT], errors="coerce")

    # Extract cancers/stages
    if COL_NOTE in df.columns:
        df["CancerTypesFromNote"] = df[COL_NOTE].apply(extract_cancers)
        df["StageFromNote"] = df[COL_NOTE].apply(extract_stage)
    else:
        df["CancerTypesFromNote"] = [[] for _ in range(len(df))]
        df["StageFromNote"] = None

    # Processing times
    if COL_REQ in df.columns and COL_SENT in df.columns:
        lag = (df[COL_SENT] - df[COL_REQ]).dt.days
        df["ProcessingDays"] = lag
        df["ProcessingDaysClean"] = np.where(lag.ge(0) & lag.le(120), lag, np.nan)

    if COL_CREATED in df.columns and COL_SENT in df.columns:
        j2s = (df[COL_SENT] - df[COL_CREATED]).dt.days
        df["JournalToSentDays"] = j2s
        df["JournalToSentDaysClean"] = np.where(j2s.ge(0) & j2s.le(120), j2s, np.nan)

    return df

df = load_data()

# ---------------- Cards ----------------
cards = []

# 1) Distribution of Cancer Types
def _card_cancer_types():
    flat = [t for lst in df["CancerTypesFromNote"] for t in (lst if isinstance(lst, list) else [])]
    if not flat:
        st.info("No cancer types could be inferred from Note."); return
    counts = pd.Series(flat).value_counts().reset_index()
    counts.columns = ["Cancer Type", "Count"]
    fig = px.bar(counts, x="Cancer Type", y="Count", text="Count", color_discrete_sequence=C1)
    fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color="#000"))
    style_fig(fig, x_title="Cancer Type", y_title="Count")
    fig.update_xaxes(showticklabels=True, tickangle=-35, tickfont=dict(color="#000", size=11),
                     automargin=True, categoryorder="total descending")
    st.plotly_chart(fig, use_container_width=True)
cards.append(("Distribution of Cancer Types", _card_cancer_types))

# 2) Satisfaction Ratings
if COL_Q1_SAT in df.columns:
    def _card_sat():
        fig = px.histogram(df, x=COL_Q1_SAT, nbins=10, color_discrete_sequence=C4)
        style_fig(fig, x_title="Satisfaction (1–10)", y_title="Responses")
        st.plotly_chart(fig, use_container_width=True)
    cards.append(("Satisfaction Ratings", _card_sat))

# 3) Items Found Most Valuable (counts)
if COL_Q2_ITEM in df.columns:
    def _card_mvi_counts():
        s = df[COL_Q2_ITEM].replace("", np.nan).dropna()
        counts = s.value_counts().reset_index()
        counts.columns = ["Item","Count"]
        fig = px.bar(counts, x="Item", y="Count", text="Count",
                     color="Item", color_discrete_sequence=MULTI)
        fig.update_traces(textposition="outside", textfont=dict(color="#000"))
        style_fig(fig, x_title="Item", y_title="Count")
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
    cards.append(("Items Found Most Valuable", _card_mvi_counts))

# 4) Awareness (bucketed with explicit sources)
if COL_AWARE in df.columns:
    def _card_awareness():
        cats = df[COL_AWARE].astype(str).apply(categorize_awareness)
        counts = (cats.value_counts().reindex(AWARE_ORDER).fillna(0).astype(int).reset_index())
        counts.columns = ["Category","Count"]
        fig = px.bar(counts, y="Category", x="Count", orientation="h",
                     text="Count", color="Category", color_discrete_sequence=MULTI2)
        fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color="#000"))
        style_fig(fig, x_title="Responses", y_title=None)
        fig.update_layout(bargap=0.25)
        st.plotly_chart(fig, use_container_width=True)
    cards.append(("How did you find out about Cancer Card", _card_awareness))

# 5) Preferred Additions / Changes — BAR ONLY with "No change" %
if COL_Q4_CHANGE in df.columns:
    def _card_changes():
        s = df[COL_Q4_CHANGE].astype(str)
        valid = s[s.str.len() > 0]
        if valid.empty:
            st.info("No responses for additions/changes."); return

        no_mask = valid.apply(is_no_change)
        n_no  = int(no_mask.sum())
        n_tot = int(valid.shape[0])

        sugg   = valid[~no_mask]
        mapped = sugg.apply(map_concise_suggestion)

        counts = mapped.value_counts().rename_axis("Suggestion").reset_index(name="Count")
        examples = (pd.DataFrame({"Suggestion": mapped, "Raw": sugg})
                    .groupby("Suggestion")["Raw"].apply(lambda x: " • ".join(x.astype(str).head(3)))
                    .reset_index().rename(columns={"Raw": "Examples"}))
        counts = counts.merge(examples, on="Suggestion", how="left")
        counts["Percent"] = (counts["Count"] / max(1, n_tot) * 100).round(1)

        main = counts[counts["Suggestion"] != "Other (misc)"].copy()
        no_row = pd.DataFrame([{
            "Suggestion": "No change",
            "Count": n_no,
            "Percent": round(n_no / max(1, n_tot) * 100, 1),
            "Examples": ""
        }])
        main = pd.concat([main, no_row], ignore_index=True)

        main = main.sort_values("Percent", ascending=True)
        main["label_txt"] = main["Count"].astype(int).astype(str) + " (" + main["Percent"].astype(str) + "%)"

        fig = px.bar(
            main, y="Suggestion", x="Percent", orientation="h",
            text="label_txt", color="Suggestion",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hover_data={"Examples": True, "Count": True, "Percent": True}
        )
        fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color="#000"))
        style_fig(fig, x_title="Share of all responses (%)", y_title=None, height=520)
        fig.update_xaxes(range=[0, 100], tickfont=dict(color="#000", size=12))
        fig.update_yaxes(tickfont=dict(color="#000", size=13), automargin=True)
        st.plotly_chart(fig, use_container_width=True)
    cards.append(("Preferred Additions / Changes", _card_changes))

# 6) Requested vs Sent — Lag Analysis (colored by lag)
if COL_REQ in df.columns and COL_SENT in df.columns:
    def _card_lag_scatter():
        lag_df = df[[COL_REQ, COL_SENT]].dropna().copy()
        if lag_df.empty:
            st.info("Not enough date pairs to plot."); return
        lag_df["lag_days"] = (lag_df[COL_SENT] - lag_df[COL_REQ]).dt.days

        def bucket(d):
            if d < 0:   return "Negative (data?)"
            if d <= 7:  return "≤7d"
            if d <= 21: return "8–21d"
            return ">21d"
        lag_df["bucket"] = lag_df["lag_days"].apply(bucket)

        fig = px.scatter(
            lag_df, x=COL_REQ, y=COL_SENT, color="bucket", opacity=0.95,
            hover_data={"lag_days": True},
            labels={"lag_days": "Lag (days)"},
            color_discrete_map={"≤7d": "#22c55e","8–21d": "#f59e0b",">21d": "#ef4444","Negative (data?)": "#8b5cf6"},
        )
        dmin = min(lag_df[COL_REQ].min(), lag_df[COL_SENT].min())
        dmax = max(lag_df[COL_REQ].max(), lag_df[COL_SENT].max())
        fig.add_trace(go.Scatter(x=[dmin, dmax], y=[dmin, dmax], mode="lines",
                                 line=dict(color="#6b7280", dash="dash"),
                                 hoverinfo="skip", showlegend=False))
        style_fig(fig, x_title="Requested (date)", y_title="Sent (date)")
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})
    cards.append(("Requested vs Sent Dates — Lag Analysis", _card_lag_scatter))

# 7) Processing Time Histogram — cleaned 0–120 days
if "ProcessingDaysClean" in df.columns:
    def _card_proc_hist():
        clean = pd.Series(df["ProcessingDaysClean"].dropna())
        if clean.empty:
            st.info("No valid processing times in 0–120 day window."); return
        fig = px.histogram(clean, x=clean, nbins=20,
                           labels={'x': 'Processing Time (days)'},
                           color_discrete_sequence=C5)
        style_fig(fig, x_title="Processing Time (days)", y_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    cards.append(("Distribution of Processing Times (Requested → Sent)", _card_proc_hist))

# 8) Journal → Sent — Monthly Distribution (box + mean line)
if "JournalToSentDaysClean" in df.columns and COL_CREATED in df.columns:
    def _card_journal_sent_box():
        d = df[[COL_CREATED, "JournalToSentDaysClean"]].dropna().copy()
        if d.empty:
            st.info("No valid Journal→Sent pairs in 0–120 day window."); return
        d["Month"] = d[COL_CREATED].dt.to_period("M").dt.to_timestamp()
        fig = px.box(d, x="Month", y="JournalToSentDaysClean",
                     color_discrete_sequence=C6, points=False)
        means = d.groupby("Month")["JournalToSentDaysClean"].mean().reset_index()
        fig.add_trace(go.Scatter(x=means["Month"], y=means["JournalToSentDaysClean"],
                                 mode="lines+markers", line=dict(color="#111", width=2),
                                 marker=dict(size=6), hovertemplate="Mean %{y:.1f}d<extra></extra>"))
        style_fig(fig, x_title="Month", y_title="Days")
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    cards.append(("Journal → Sent — Monthly Distribution", _card_journal_sent_box))

# 9) Recommendation — Yes vs No (%)
if COL_Q6_RECO in df.columns:
    def _card_reco_yes_no():
        s = df[COL_Q6_RECO].astype(str).str.strip().str.lower()
        s = s.map(lambda v: "Yes" if v.startswith("yes") else ("No" if v.startswith("no") else np.nan)).dropna()
        if s.empty:
            st.info("No recommendation data available."); return
        counts = s.value_counts().reindex(["Yes","No"]).fillna(0).astype(int)
        total = int(counts.sum())
        plot_df = pd.DataFrame({"Response": ["Yes","No"],
                                "Percent": [round(counts["Yes"]/total*100,1) if total else 0.0,
                                            round(counts["No"]/total*100,1) if total else 0.0]})
        fig = px.bar(plot_df, x="Response", y="Percent", text="Percent",
                     color="Response", color_discrete_sequence=MULTI2)
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside",
                          cliponaxis=False, textfont=dict(color="#000"))
        style_fig(fig, x_title="Response", y_title="Percent")
        fig.update_yaxes(range=[0,100])
        st.plotly_chart(fig, use_container_width=True)
    cards.append(("Recommend the Comfort Box — Yes vs No (%)", _card_reco_yes_no))

# 10) Comfort Box Requests Over Time
if COL_REQ in df.columns:
    def _card_monthly_requests():
        tmp = df[[COL_REQ]].dropna().copy()
        tmp["Month"] = tmp[COL_REQ].dt.to_period("M").dt.to_timestamp()
        counts = tmp["Month"].value_counts().sort_index().reset_index()
        counts.columns = ["Month","Number of Requests"]
        fig = px.bar(counts, x="Month", y="Number of Requests",
                     text="Number of Requests", color_discrete_sequence=C4)
        fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color="#000"))
        style_fig(fig, x_title="Month", y_title="Number of Requests")
        st.plotly_chart(fig, use_container_width=True)
    cards.append(("Comfort Box Requests Over Time", _card_monthly_requests))
if COL_Q2_ITEM in df.columns:
    def _card_mvi_pct():
        s = df[COL_Q2_ITEM].replace("", np.nan).dropna()
        total = int(s.shape[0])
        if total == 0: st.info("No data."); return
        cnt = s.value_counts()
        data = (pd.DataFrame({"Item": cnt.index, "Count": cnt.values,
                              "Percent": (cnt/total*100).round(1)})
                  .sort_values("Count", ascending=True))
        fig = px.bar(data, y="Item", x="Count", orientation="h",
                     text=data["Percent"].astype(str) + "%", color="Item",
                     color_discrete_sequence=MULTI)
        fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color="#000"))
        style_fig(fig, x_title="Responses", y_title=None)
        st.plotly_chart(fig, use_container_width=True)
    cards.append(("Most Valuable Comfort Box Items (with %)", _card_mvi_pct))

def _card_cancer_stage_overall():
    rows = []
    for _, r in df.iterrows():
        types = r.get("CancerTypesFromNote", [])
        if not types: continue
        stage = r.get("StageFromNote") or "Stage (unspecified)"
        for t in types: rows.append(f"{t} — {stage}")
    if not rows:
        st.info("No cancer type/stage mentions detected."); return
    s = pd.Series(rows).value_counts().head(20).sort_values(ascending=True).reset_index()
    s.columns = ["Cancer — Stage", "Count"]
    fig = px.bar(s, y="Cancer — Stage", x="Count", orientation="h", color_discrete_sequence=C8)
    style_fig(fig, x_title="Count", y_title=None)
    st.plotly_chart(fig, use_container_width=True)
cards.append(("Top Cancer–Stage Mentions (overall)", _card_cancer_stage_overall))

# 13–15) Met expectations suite
if COL_Q3_EXPECT in df.columns:
    exp_series = df[COL_Q3_EXPECT].astype(str)
    met_mask   = exp_series.str.strip().str.lower().str.startswith(("yes","partial"))
    valid_mask = exp_series.str.len() > 0
    rate = (met_mask.sum() / max(1, valid_mask.sum()))

    def _card_overall_bullet():
        pct = rate * 100.0
        fig = go.Figure()
        fig.add_bar(y=[""], x=[pct],       orientation="h", marker_color=C6[0], hoverinfo="skip")
        fig.add_bar(y=[""], x=[100 - pct], orientation="h", marker_color="#e5e7eb", hoverinfo="skip")
        fig.add_scatter(y=[""], x=[pct], mode="markers+text",
                        marker=dict(size=10, color="#111"),
                        text=[f"{pct:.1f}%"], textposition="top center",
                        hoverinfo="skip", showlegend=False)
        style_fig(fig, showgrid=False, height=160,
                  title_text=f"Met expectations (overall): {pct:.1f}%",
                  x_title=None, y_title=None)
        fig.update_xaxes(range=[0, 100]); fig.update_yaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
    cards.append(("Met expectations (overall)", _card_overall_bullet))

    def _card_expect_by_cancer():
        records = []
        for _, row in df.iterrows():
            types = row.get("CancerTypesFromNote", [])
            if not isinstance(types, list) or not types: continue
            met = is_yes_or_partial(row.get(COL_Q3_EXPECT, ""))
            for t in types: records.append((t, met))
        if not records:
            st.info("No cancer types extracted to compute this chart."); return
        tmp = pd.DataFrame(records, columns=["Cancer","Met"])
        agg = tmp.groupby("Cancer")["Met"].agg(["sum","count"]).query("count>=5")
        if agg.empty: st.info("No cancer type with at least 5 responses."); return
        out = (agg.assign(rate=lambda d: d["sum"]/d["count"])
                 .sort_values("rate", ascending=True).reset_index())
        fig = px.bar(out, y="Cancer", x=(out["rate"]*100), orientation="h",
                     text=(out["rate"]*100).round(0).astype(int).astype(str)+"%",
                     color_discrete_sequence=C9)
        fig.update_traces(textposition="outside", textfont=dict(color="#000"))
        style_fig(fig, x_title="Share met (%)", y_title=None)
        fig.update_xaxes(range=[0,100])
        st.plotly_chart(fig, use_container_width=True)
    cards.append(("Met Expectations by Cancer Type (min n=5)", _card_expect_by_cancer))

    if COL_REQ in df.columns:
        def _card_expect_trend():
            tmp = df[[COL_REQ, COL_Q3_EXPECT]].dropna(subset=[COL_REQ]).copy()
            tmp["month"] = tmp[COL_REQ].dt.to_period("M").dt.to_timestamp()
            tmp["met"] = tmp[COL_Q3_EXPECT].astype(str).str.lower().str.startswith(("yes","partial"))
            agg = tmp.groupby("month").agg(n=("met","size"), k=("met","sum"))
            agg["rate"] = agg["k"]/agg["n"]
            plot = agg.reset_index()
            fig = px.line(plot, x="month", y=(plot["rate"]*100), markers=True,
                          color_discrete_sequence=C1)
            style_fig(fig, x_title="Month", y_title="Share met (%)")
            fig.update_yaxes(range=[0,100])
            st.plotly_chart(fig, use_container_width=True)
        cards.append(("Met Expectations — Monthly trend", _card_expect_trend))

if COL_Q5_EMO in df.columns:
    @st.cache_resource(show_spinner=False)
    def get_vader():
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            import nltk
            try:
                _ = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download('vader_lexicon')
            return SentimentIntensityAnalyzer()
        except Exception:
            return None

    def _card_sentiment():
        texts = [t for t in df[COL_Q5_EMO].astype(str) if t.strip()]
        if len(texts) == 0:
            st.info("No emotional well-being text to analyse."); return
        sia = get_vader()
        if sia is None:
            pos_words = ["good","great","helpful","comfort","happy","support","love","nice","positive","relief","calm"]
            neg_words = ["bad","poor","sad","upset","stress","anxious","angry","negative","worse","pain","tired"]
            def quick_sent(s):
                t = s.lower()
                if any(w in t for w in neg_words) and not any(w in t for w in pos_words): return "Negative"
                if any(w in t for w in pos_words) and not any(w in t for w in neg_words): return "Positive"
                return "Neutral"
            labels = [quick_sent(t) for t in texts]
        else:
            def lab(score):
                if score >= 0.2: return "Positive"
                if score <= -0.2: return "Negative"
                return "Neutral"
            labels = [lab(sia.polarity_scores(t)["compound"]) for t in texts]

        s = pd.Series(labels)
        counts = s.value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0).astype(int)
        total = counts.sum() if counts.sum() else 1
        dfp = (counts / total * 100).round(1).reset_index()
        dfp.columns = ["Sentiment","Percent"]

        fig = px.bar(dfp, y="Sentiment", x="Percent", orientation="h",
                     text="Percent", color="Sentiment", color_discrete_sequence=MULTI2)
        fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside",
                          cliponaxis=False, textfont=dict(color="#000"))
        style_fig(fig, x_title="Share of responses (%)", y_title=None)
        fig.update_xaxes(range=[0,100])
        st.plotly_chart(fig, use_container_width=True)
    cards.append(("Emotional Well-Being — Sentiment Analysis", _card_sentiment))
rows = (len(cards) + GRID_COLS - 1) // GRID_COLS
idx = 0
for _ in range(rows):
    cols = st.columns(GRID_COLS, gap="large")
    for c in cols:
        if idx >= len(cards): break
        title, fn = cards[idx]
        put_card(c, title, fn)
        idx += 1
