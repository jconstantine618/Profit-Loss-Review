# profit_loss_review.py  (partial‑month aware) ──────────────────────
import os, re, tempfile, calendar
from datetime import datetime

import numpy as np
import openai
import pandas as pd
import streamlit as st
from gtts import gTTS

st.set_page_config(page_title="📊 Monthly P&L Financial Review", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

MONTH_RE = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$", re.I)


def load_pl(file) -> pd.DataFrame:
    raw = pd.read_excel(file, header=None, engine="openpyxl")

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
        month_name, year = m.group(1), int(str(current_col).split()[-1])
        month_num = datetime.strptime(month_name[:3], "%b").month
        if month_num == today.month and year == today.year:
            partial = True
            days_in_month = calendar.monthrange(year, month_num)[1]
            pct_complete = today.day / days_in_month

    compare_col = numeric_cols[-2] if len(numeric_cols) >= 2 else None
    return current_col, compare_col, partial, pct_complete


def build_summary(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include="number").columns

    if {"Actual", "Budget"}.issubset(num_cols):
        current_col, compare_col = "Actual", "Budget"
        partial, pct_complete = False, 1.0
    else:
        current_col, compare_col, partial, pct_complete = detect_partial_month(num_cols)

    df_work = df[["Category"] + list(num_cols)].copy()

    if compare_col:
        df_work["Variance"] = df_work[current_col] - df_work[compare_col]
        df_work["Variance %"] = (
            df_work["Variance"] / df_work[compare_col].replace({0: np.nan}) * 100
        )
        key_rows = df_work[df_work["Variance %"].abs() >= 10]
    else:
        key_rows = df_work.copy()

    display_cols = ["Category", current_col]
    if compare_col:
        display_cols += [compare_col, "Variance", "Variance %"]
    table_txt = key_rows[display_cols].to_string(index=False)

    partial_note = (
        f"\n⚠️  **Note:** {current_col} is month‑to‑date "
        f"({pct_complete:.0%} of the month complete). "
        "Discuss results as provisional and emphasise run‑rate projections.\n"
        if partial
        else ""
    )

    prompt = f"""
You are a CFO preparing a month‑end (or MTD) briefing for the CEO.

{partial_note}
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
I’ll detect if the latest month is MTD, give you a CFO‑style summary, optional
audio, and a follow‑up Q&A panel.
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

if "summary" in st.session_state and st.button("🔊 Listen to Summary"):
    play_text(st.session_state["summary"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("💬 Ask a follow‑up question")
user_q = st.text_input("Question")

if st.button("Ask") and user_q:
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a CFO answering questions about the company's P&L."},
            {"role": "user", "content": user_q},
        ],
    )
    answer = resp.choices[0].message.content.strip()
    st.session_state.chat_history.append((user_q, answer))

for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
