import os
import streamlit as st
import pandas as pd
import openai
from gtts import gTTS

# ---------- CONFIG ----------
st.set_page_config(page_title="📊 Monthly P&L Financial Review", layout="wide")

# Read API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---------- PAGE HEADER ----------
st.title("📊 Monthly P&L Financial Review")
st.markdown(
    "Upload an **Excel** Profit & Loss statement with ‘Actual’ and ‘Budget’ columns. "
    "I’ll create a CFO‑style summary and a playable audio snippet, then you can ask follow‑up questions. "
)

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("📁 Upload P&L Excel File (.xlsx)", type=["xlsx"])

# ---------- FUNCTIONS ----------
def build_summary(df: pd.DataFrame) -> str:
    """Return a board‑meeting‑style CFO summary (≈5‑min script)."""
    # Clean / calc variance
    df = df.dropna(subset=["Actual", "Budget"]).copy()
    df["Variance"] = df["Actual"] - df["Budget"]
    df["Variance %"] = df["Variance"] / df["Budget"] * 100

    # Focus on big movers (> ±10 %)
    key_rows = df[abs(df["Variance %"]) >= 10]
    table_txt = key_rows.to_string(index=False)

    prompt = f"""
You are a CFO summarizing monthly financial results for the CEO.

P&L items with >10% variance vs. budget:
{table_txt}

Give a confident, plain‑English narrative (no more than ~5 minutes read aloud). 
Highlight revenue trends, margin pressure, cost overruns/savings, and cash‑flow implications. 
Close with two or three action items for next month.
"""

    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

def play_text(text: str):
    """Generate TTS and stream back to user."""
    tts = gTTS(text)
    mp3_path = "summary_audio.mp3"
    tts.save(mp3_path)
    with open(mp3_path, "rb") as f:
        st.audio(f.read(), format="audio/mp3")
    os.remove(mp3_path)

# ---------- MAIN ----------
if uploaded_file:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    st.subheader("📄 P&L Preview")
    st.dataframe(df)

    if st.button("📊 Generate CFO Summary"):
        summary = build_summary(df)
        st.session_state["summary"] = summary
        st.text_area("📝 CFO Summary Script", summary, height=300)

# ---------- LISTEN BUTTON ----------
if "summary" in st.session_state:
    if st.button("🔊 Listen to Summary"):
        play_text(st.session_state["summary"])

# ---------- Q&A CHAT ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("💬 Ask the CFO a question")
user_q = st.text_input("Your question")

if st.button("Ask") and user_q:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a CFO answering questions about a company's P&L."},
            {"role": "user", "content": user_q},
        ],
    )
    answer = response.choices[0].message.content.strip()
    st.session_state.chat_history.append((user_q, answer))

# Show history
for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
