# profit_loss_review.py  ────────────────────────────────────────────
"""
Monthly P&L Financial Review (Streamlit + OpenAI + optional TTS)

• Upload a QuickBooks “P&L by Month” export  ── or ──  any sheet that has
  “Actual” and “Budget” columns.
• The app detects partial‑month data (MTD), highlights ≥10 % variances,
  asks GPT‑4o for a 400‑–600‑word CFO‑style narrative, and (optionally) reads
  it aloud.
• If a “YTD / Total” column exists, the narrative includes a Year‑to‑Date view.
• A plain‑English disclaimer is shown beneath the summary.

Requirements:  streamlit, pandas, numpy, openai, gTTS, openpyxl
"""

from __future__ import annotations

import calendar
import os
import re
import tempfile
from datetime import datetime

import numpy as np
import openai
import pandas as pd
import streamlit as st
from gtts import gTTS

# ── CONFIG ─────────────────────────────────────────────────────────
st.set_page_config(page_title="📊 Monthly P&L Financial Review", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

MONTH_RE = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$", re.I)
YTD_RE = re.compile(r"\b(YTD|Year\s*to\s*Date|Total)\b", re.I)


# ── HELPERS ────────────────────────────────────────────────────────
def load_pl(file) -> pd.DataFrame:
    """Load a QuickBooks ‘P&L by Month’ OR a simple Actual/Budget sheet."""
    # Read raw with no header so we can locate the month header row.
    raw = pd.read_excel(file, header=None, engine="openpyxl")

    # Find the first row that contains at least three recognised month labels.
    header_idx = next(
        (i for i, row in raw.iterrows() if sum(bool(MONTH_RE.match(str(c))) for c in row) >= 3),
        0,
    )

    df = pd.read_excel(file, header=header_idx, engine="openpyxl")
    df = df.rename(columns={df.columns[0]: "Category"})
    df = df.dropna(how="all").copy()
    df["Category"] = df["Category"].ffill()

    return df


def detect_partial_month(numeric_cols):
    """Return (current_col, compare_col, partial_flag, pct_complete)."""
    today = datetime.today()
    current_col = numeric_cols[-1]

    m = MONTH_RE.match(str(current_col))
    partial = False
    pct_complete = 1.0

    if m:
        # Parse the month and year from the column label, e.g. "May 2025"
        month_name, year = m.group(1), int(str(current_col).split()[-1])
        month_num = datetime.strptime(month_name[:3], "%b").month

        if month_num == today.month and year == today.year:
            partial = True
            days_in_month = calendar.monthrange(year, month_num)[1]
            pct_complete = today.day / days_in_month

    compare_col = numeric_cols[-2] if len(numeric_cols) >= 2 else None
    return current_col, compare_col, partial, pct_complete


def build_summary(df: pd.DataFrame) -> str:
    """Return a ≈ 5‑minute, board‑ready CFO narrative."""
    num_cols = df.select_dtypes(include="number").columns

    # ── Determine current & comparison columns ─────────────────────
    if {"Actual", "Budget"}.issubset(num_cols):
        current_col, compare_col = "Actual", "Budget"
        partial, pct_complete = False, 1.0
    else:
        current_col, compare_col, partial, pct_complete = detect_partial_month(num_cols)

    # ── Identify a YTD / Total column if present ───────────────────
    ytd_cols = [c for c in num_cols if YTD_RE.search(str(c))]
    ytd_col = ytd_cols[-1] if ytd_cols else None

    # ── Variance calculation ───────────────────────────────────────
    df_work = df[["Category"] + list(num_cols)].copy()

    if compare_col:
        df_work["Variance"] = df_work[current_col] - df_work[compare_col]
        df_work["Variance %"] = (
            df_work["Variance"] / df_work[compare_col].replace({0: np.nan}) * 100
        )
        key_rows = df_work[df_work["Variance %"].abs() >= 10]
    else:
        key_rows = df_work.copy()

    # ── Build plain‑text table for GPT ‑ We only send the key rows ──
    display_cols = ["Category", current_col]
    if compare_col:
        display_cols += [compare_col, "Variance", "Variance %"]
    table_txt = key_rows[display_cols].to_string(index=False)

    # ── Prompt components ──────────────────────────────────────────
    partial_note = ""
    if partial:
        partial_note = (
            f"⚠️  **Note:** {current_col} is month‑to‑date "
            f"({pct_complete:.0%} of the month complete). "
            "This is **not a complete snapshot of the month**; treat variances "
            "as provisional and focus on run‑rate projections.\n"
        )

    ytd_note = (
        f"\nAlso provide a concise perspective on **Year‑to‑Date performance** using the "
        f"`{ytd_col}` column."
        if ytd_col
        else ""
    )

    prompt = f"""
You are a CFO preparing a month‑end (or MTD) briefing for the CEO.

{partial_note}{ytd_note}

Here are the key line‑items (≥10 % variance) comparing **{current_col}**
with **{compare_col or 'Prior Period'}**:

{table_txt}

Write ~400‑600 words:
• Revenue & gross‑margin movement  
• Major cost drivers / savings  
• Cash‑flow or balance‑sheet implications  
• 2‑3 action items for next month (or for the rest of the current month if MTD).

Avoid jargon, be decisive, and clearly flag any MTD caveats.
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def play_text(text: str):
    """Generate a temporary MP3 with gTTS and stream it in‑app."""
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        st.audio(open(tmp.name, "rb").read(), format="audio/mp3")
    os.remove(tmp.name)


# ── UI ────────────────────────────────────────────────────────────
st.title("📊 Monthly P&L Financial Review")

st.markdown(
    """
Upload an **Excel** Profit & Loss statement (QuickBooks “by Month” export or a
sheet with “Actual” and “Budget” columns).  
The app will detect if the latest month is MTD, highlight ≥10 % variances,
and generate a CFO‑style narrative.  
""",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("📁 Upload P&L (.xlsx)", type="xlsx")

if uploaded:
    df = load_pl(uploaded)
    st.subheader("📄 P&L Preview")
    st.dataframe(df, use_container_width=True)

    if st.button("📝 Generate CFO Summary"):
        summary = build_summary(df)
        st.session_state["summary"] = summary
        st.text_area("CFO Summary", summary, height=320)

        # ── Disclaimer (always visible after summary) ────────────
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
            of this review. As a best practice, always examine your financial statements in
            depth and consult a qualified accounting or finance professional before acting.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Listen to summary ─────────────────────────────────────────────
if "summary" in st.session_state and st.button("🔊 Listen to Summary"):
    play_text(st.session_state["summary"])

# ── Q&A panel ─────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("💬 Ask a follow‑up question")
user_q = st.text_input("Question")

if st.button("Ask") and user_q:
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a CFO answering questions about the company's P&L.",
            },
            {"role": "user", "content": user_q},
        ],
    )
    answer = resp.choices[0].message.content.strip()
    st.session_state.chat_history.append((user_q, answer))

# Display chat history in reverse chronological order
for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
