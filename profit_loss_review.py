# profit_loss_review.py  ────────────────────────────────────────────
import os
import re
import tempfile

import numpy as np
import openai
import pandas as pd
import streamlit as st
from gtts import gTTS

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(page_title="📊 Monthly P&L Financial Review", layout="wide")

# ── OPENAI KEY ────────────────────────────────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ── HELPERS ───────────────────────────────────────────────────────
MONTH_RE = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", re.I)


def load_pl(file) -> pd.DataFrame:
    """Load a QuickBooks ‘P&L by Month’ OR simple Actual/Budget sheet."""
    # read raw with no header
    raw = pd.read_excel(file, header=None, engine="openpyxl")

    # find the first row that looks like the month header
    header_idx = None
    for idx, row in raw.iterrows():
        if sum(bool(MONTH_RE.search(str(cell))) for cell in row) >= 3:
            header_idx = idx
            break

    if header_idx is None:  # fall back to first row as header
        df = pd.read_excel(file, engine="openpyxl")
    else:
        df = pd.read_excel(file, header=header_idx, engine="openpyxl")

    # normalise first (category) column name
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Category"})

    # drop completely empty rows & forward‑fill category hierarchy
    df = df.dropna(how="all").copy()
    df["Category"] = df["Category"].ffill()

    return df


def build_summary(df: pd.DataFrame) -> str:
    """Return a ≈5‑minute, board‑ready CFO narrative."""
    num_cols = df.select_dtypes(include="number").columns

    # If an Actual/Budget layout exists, use it; otherwise pick last two months
    if {"Actual", "Budget"}.issubset(num_cols):
        current_col, compare_col = "Actual", "Budget"
    else:
        # ignore a trailing 'Total' column if present
        candidates = [c for c in num_cols if "total" not in str(c).lower()]
        candidates = candidates or list(num_cols)  # fallback
        current_col = candidates[-1]
        compare_col = candidates[-2] if len(candidates) >= 2 else None

    df_work = df[["Category"] + num_cols.tolist()].copy()

    if compare_col:
        df_work["Variance"] = df_work[current_col] - df_work[compare_col]
        df_work["Variance %"] = (
            df_work["Variance"] / df_work[compare_col].replace({0: np.nan}) * 100
        )
        key_rows = df_work[df_work["Variance %"].abs() >= 10]
    else:
        key_rows = df_work.copy()

    # format a small text table for GPT
    display_cols = ["Category", current_col]
    if compare_col:
        display_cols += [compare_col, "Variance", "Variance %"]
    table_txt = key_rows[display_cols].to_string(index=False)

    prompt = f"""
You are a CFO preparing a month‑end briefing for the CEO.

Here are the line items with >10% variance (or the most material figures) comparing **{current_col}** with **{compare_col or 'Prior Period'}**:

{table_txt}

Craft a confident, plain‑English narrative (≈400‑600 words, ~5 min read aloud).  
Highlight: revenue trends, margins, cost overruns / savings, cash‑flow implications.  
Close with 2‑3 clear action items for next month. Avoid jargon.
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def play_text(text: str):
    """Generate TTS, stream it, then clean up."""
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    with open(tmp.name, "rb") as f:
        st.audio(f.read(), format="audio/mp3")
    os.remove(tmp.name)


# ── UI ────────────────────────────────────────────────────────────
st.title("📊 Monthly P&L Financial Review")

st.markdown(
    """
Upload an **Excel** Profit & Loss statement (QuickBooks “by Month” export or a
sheet with “Actual” and “Budget” columns).  
I’ll give you a CFO‑style summary & an optional audio playback, then you can ask follow‑up questions.
"""
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

# — LISTEN —
if "summary" in st.session_state:
    if st.button("🔊 Listen to Summary"):
        play_text(st.session_state["summary"])

# — Q&A —
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

for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
