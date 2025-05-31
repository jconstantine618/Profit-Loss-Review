# financial_review_app.py  ─────────────────────────────────────────
"""
Financial Review  (P&L + optional Balance Sheet)   •  Streamlit + OpenAI + gTTS

Deploy‑ready for Render.com
──────────────────────────
• Set the environment variable OPENAI_API_KEY in Render.
• Start command:
    streamlit run financial_review_app.py --server.port $PORT --server.address 0.0.0.0
"""

from __future__ import annotations

import calendar
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import openai
import pandas as pd
import streamlit as st
from gtts import gTTS

# ── CONFIG ─────────────────────────────────────────────────────────
st.set_page_config(page_title="📊 Financial Review (P&L + B/S)", layout="wide")

# Read the OpenAI key from an environment variable (Render‑friendly)
openai.api_key = os.getenv("OPENAI_API_KEY", "")  # ← add your key in Render dashboard

MONTH_RE = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$", re.I)
YTD_RE   = re.compile(r"\b(YTD|Year\s*to\s*Date|Total)\b", re.I)

# ── Loaders ────────────────────────────────────────────────────────
def _read_any(file, **kwargs) -> pd.DataFrame:
    ext = Path(file.name).suffix.lower()
    if ext in {".csv", ".txt"}:
        return pd.read_csv(file, **kwargs)
    return pd.read_excel(file, engine="openpyxl", **kwargs)


def load_pl(file) -> pd.DataFrame:
    """Return a cleaned P&L DataFrame. First column becomes 'Category'."""
    raw = _read_any(file, header=None)
    header_idx = next(
        (i for i, row in raw.iterrows() if sum(bool(MONTH_RE.match(str(c))) for c in row) >= 3), 0
    )
    df = _read_any(file, header=header_idx)
    df = df.rename(columns={df.columns[0]: "Category"})
    df = df.dropna(how="all").copy()
    df["Category"] = df["Category"].ffill()
    return df


def load_bs(file) -> pd.DataFrame:
    """Load a basic Balance Sheet (rows = accounts, first numeric col = current)."""
    df = _read_any(file)
    df = df.rename(columns={df.columns[0]: "Account"})
    df = df.dropna(how="all").copy()
    df["Account"] = df["Account"].ffill()
    return df


# ── Helpers ────────────────────────────────────────────────────────
def detect_partial_month(num_cols):
    """Return (current_col, compare_col, is_partial, pct_complete)."""
    today = datetime.today()
    current_col = num_cols[-1]

    m = MONTH_RE.match(str(current_col))
    partial, pct_complete = False, 1.0
    if m:
        month_name, year = m.group(1), int(str(current_col).split()[-1])
        month_num = datetime.strptime(month_name[:3], "%b").month
        if month_num == today.month and year == today.year:
            partial = True
            days_in_month = calendar.monthrange(year, month_num)[1]
            pct_complete = today.day / days_in_month
    compare_col = num_cols[-2] if len(num_cols) >= 2 else None
    return current_col, compare_col, partial, pct_complete


def get_row_val(df: pd.DataFrame, keywords) -> Optional[float]:
    mask = df.iloc[:, 0].str.lower().str.contains("|".join(keywords), na=False)
    numeric_cols = df.select_dtypes(include="number").columns
    if mask.any() and len(numeric_cols):
        return df.loc[mask, numeric_cols[0]].values[0]
    return None


# ── Narrative builders ────────────────────────────────────────────
def build_pl_summary(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include="number").columns

    if {"Actual", "Budget"}.issubset(num_cols):
        current_col, compare_col = "Actual", "Budget"
        partial, pct_complete = False, 1.0
    else:
        current_col, compare_col, partial, pct_complete = detect_partial_month(num_cols)

    ytd_cols = [c for c in num_cols if YTD_RE.search(str(c))]
    ytd_col = ytd_cols[-1] if ytd_cols else None

    df_work = df[["Category"] + list(num_cols)].copy()
    if compare_col:
        df_work["Variance"] = df_work[current_col] - df_work[compare_col]
        df_work["Variance %"] = (
            df_work["Variance"] / df_work[compare_col].replace({0: np.nan}) * 100
        )

    key_rows = df_work if not compare_col else df_work[df_work["Variance %"].abs() >= 10]

    cols_for_table = ["Category", current_col]
    if compare_col:
        cols_for_table += [compare_col, "Variance", "Variance %"]
    if ytd_col:
        cols_for_table.append(ytd_col)
    table_txt = key_rows[cols_for_table].to_string(index=False)

    partial_note = (
        f"⚠️  **Note:** {current_col} is month‑to‑date "
        f"({pct_complete:.0%} complete). Not a full snapshot.\n"
        if partial else ""
    )
    ytd_note = f"\nThe **{ytd_col}** column shows Year‑to‑Date performance." if ytd_col else ""

    prompt = f"""
You are a CFO preparing a briefing for the CEO.

{partial_note}{ytd_note}

Key line‑items (≥10 % variance) comparing **{current_col}**
with **{compare_col or 'Prior Period'}**{(' plus YTD' if ytd_col else '')}:

{table_txt}

Write 400‑600 words covering:
• Revenue & gross‑margin movement  
• Major cost drivers or savings  
• Cash‑flow implications  
• 2‑3 clear action items.

Avoid jargon; explicitly flag any MTD caveats.
Based on the attached Profit & Loss statements (month-on-month or YTD by month) and the Balance Sheet, please provide a clear and simple analysis of the business's financial health. Focus on key takeaways that a non-finance business owner can understand. Specifically, please highlight:

Profitability trends: Are we making more or less money over time? What are the main drivers of these changes (e.g., increased sales, higher costs)?
Liquidity: Do we have enough readily available cash to meet our short-term obligations?
Overall financial position: What does the Balance Sheet tell us about our assets, liabilities, and equity? Are there any areas of concern or strength?
Key ratios (explained simply): If possible, point out 1-2 simple but important ratios (like a very basic profitability or liquidity indicator) and explain what they mean for the business.
Please avoid jargon and explain everything in plain language.
"""

    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return resp.choices[0].message.content.strip()


def build_bs_summary(df: pd.DataFrame) -> str:
    assets      = get_row_val(df, ["total assets"])
    curr_assets = get_row_val(df, ["total current assets"])
    cash        = get_row_val(df, ["cash", "checking"])
    liab        = get_row_val(df, ["total liabilities"])
    curr_liab   = get_row_val(df, ["total current liabilities"])
    debt        = get_row_val(df, ["total long term liabilities", "notes payable", "loan"])
    equity      = get_row_val(df, ["total equity", "retained earnings"])

    ratios = {}
    if curr_assets and curr_liab:
        ratios["Current Ratio"] = curr_assets / curr_liab
        ratios["Quick Ratio"]   = (cash or 0) / curr_liab
    if debt is not None and equity:
        ratios["Debt‑to‑Equity"] = debt / equity
    if assets and equity:
        ratios["Equity % of Assets"] = equity / assets * 100
    nwc = curr_assets - curr_liab if curr_assets and curr_liab else None

    ratio_lines = "\n".join(f"- **{k}**: {v:,.2f}" for k, v in ratios.items())
    if nwc is not None:
        ratio_lines += f"\n- **Net Working Capital**: {nwc:,.0f}"

    prompt = f"""
You are a CFO preparing a Balance‑Sheet overview for the CEO.

Metrics for the current period:
{ratio_lines or '- (not enough data to compute ratios)'}

Write 300‑450 words that explain:
• Liquidity position (Current & Quick ratios)  
• Leverage (Debt‑to‑Equity) & capital structure  
• Net Working Capital trend or concerns  
• 2‑3 recommendations to improve balance‑sheet health.

Refer only to the metrics above—do **not** invent numbers.
"""

    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return resp.choices[0].message.content.strip()


def play_text(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        st.audio(open(tmp.name, "rb").read(), format="audio/mp3")
    os.remove(tmp.name)


# ── Sidebar: Finance 101 ───────────────────────────────────────────
st.sidebar.markdown("## 💡 Finance 101")
with st.sidebar.expander("P&L building blocks"):
    st.markdown(
        """
* **Revenue** – Income from core operations.  
* **COGS** – Direct costs of goods/services sold.  
* **Gross Profit** – Revenue minus COGS.  
* **Fixed / Operating Costs** – Expenses that don’t vary directly with sales.  
* **EBITDA** – Earnings before Interest, Taxes, Depreciation & Amortisation.
"""
    )
with st.sidebar.expander("Balance‑Sheet basics"):
    st.markdown(
        """
* **Assets** – Resources owned or controlled by the company.  
* **Liabilities** – Obligations to outsiders.  
* **Equity** – Owners’ residual interest (Assets − Liabilities).  
* **Current Ratio** – Current Assets ÷ Current Liabilities.  
* **Quick Ratio** – (Cash + AR) ÷ Current Liabilities.  
* **Debt‑to‑Equity** – Total Debt ÷ Equity.
"""
    )

# ── UI ────────────────────────────────────────────────────────────
st.title("📊 Financial Review")

pl_file = st.file_uploader("📁 Upload P&L (.csv/.xlsx)", type=["csv", "xls", "xlsx"])
bs_file = st.file_uploader("📄 Upload Balance Sheet (optional)", type=["csv", "xls", "xlsx"])

if st.button("🔎 Analyse"):
    pl_summary = bs_summary = None

    if pl_file:
        df_pl = load_pl(pl_file)
        st.subheader("P&L Preview")
        st.dataframe(df_pl, use_container_width=True)
        pl_summary = build_pl_summary(df_pl)
        st.text_area("📄 P&L CFO Summary", pl_summary, height=320)
    else:
        st.warning("No P&L provided — skipping P&L analysis.")

    if bs_file:
        df_bs = load_bs(bs_file)
        st.subheader("Balance Sheet Preview")
        st.dataframe(df_bs, use_container_width=True)
        bs_summary = build_bs_summary(df_bs)
        st.text_area("🏛️ Balance Sheet Summary", bs_summary, height=320)
    else:
        st.info("No Balance Sheet provided — skipping B/S analysis.")

    # Disclaimer
    st.markdown(
        """
        <div style='border-top:1px solid #bbb; margin-top:1rem; padding-top:1rem;
                    font-size:0.9rem; color:#555;'>
        **Disclaimer**  
        This report was generated automatically with artificial‑intelligence tools and
        <strong>does not constitute professional financial advice</strong>. MAPL Health
        makes <strong>no warranties or representations</strong> regarding the accuracy,
        completeness, or suitability of the information for your specific circumstances.
        <strong>You remain solely responsible for any decisions</strong> made on the basis
        of this review. Always consult a qualified finance professional.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Listen buttons
    if pl_summary and st.button("🔊 Listen to P&L Summary"):
        play_text(pl_summary)
    if bs_summary and st.button("🔊 Listen to B/S Summary"):
        play_text(bs_summary)

    # Q&A
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("💬 Ask a follow‑up question")
    user_q = st.text_input("Question")

    if st.button("Ask") and user_q:
        context = [
            {
                "role": "system",
                "content": "You are a CFO answering questions about the company's P&L and Balance Sheet.",
            }
        ]
        if pl_summary:
            context.append({"role": "assistant", "content": pl_summary})
        if bs_summary:
            context.append({"role": "assistant", "content": bs_summary})
        context.append({"role": "user", "content": user_q})

        resp = openai.chat.completions.create(
            model="gpt-4o", messages=context, temperature=0.4
        )
        answer = resp.choices[0].message.content.strip()
        st.session_state.chat_history.append((user_q, answer))

    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
