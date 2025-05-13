import streamlit as st
import pandas as pd
import openai
from io import BytesIO
from datetime import datetime
from gtts import gTTS
import os

# ---- Config ----
st.set_page_config(page_title="Monthly Financial Summary", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---- UI ----
st.title("📊 Monthly P&L Financial Review")
st.markdown("Upload your Excel-based Profit & Loss statement. A CFO-style report will be generated along with an audio summary.")

uploaded_file = st.file_uploader("📁 Upload P&L Excel File", type=["xlsx"])

summary_text = ""

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📄 Preview of Uploaded File")
    st.dataframe(df.head())

    # ---- Prepare Input for GPT ----
    def generate_summary(df):
        df_clean = df.dropna(subset=["Actual", "Budget"])
        df_clean["Variance"] = df_clean["Actual"] - df_clean["Budget"]
        df_clean["Variance %"] = df_clean["Variance"] / df_clean["Budget"] * 100

        key_items = df_clean[abs(df_clean["Variance %"]) > 10].copy()
        key_items_str = key_items.to_string(index=False)

        prompt = f"""
You are a CFO summarizing the monthly financials to the CEO of a small business. Below is the P&L breakdown with actuals, budgets, and variance:

{key_items_str}

Create a concise verbal report (like a podcast), under 5 minutes, using plain language. Focus on areas where performance diverged from the budget, major cost concerns, strong revenue performance, and cash management implications. Avoid financial jargon unless necessary.
"""
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content

    if st.button("📊 Generate CFO Summary"):
        summary_text = generate_summary(df)
        st.session_state["summary_text"] = summary_text
        st.text_area("📝 CFO-Style Summary", summary_text, height=300)

if "summary_text" in st.session_state:
    if st.button("🔊 Listen to Summary"):
        tts = gTTS(st.session_state["summary_text"])
        audio_file = "summary_audio.mp3"
        tts.save(audio_file)
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")
        os.remove(audio_file)

# ---- Optional: Chat Q&A ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("💬 Ask Questions About This Month's Financials")

question = st.text_input("Type your question:")
if st.button("Ask") and question:
    chat_prompt = f"You are a CFO reviewing the company's P&L. Here's the user's question: {question}"

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a CFO answering financial questions based on a P&L."},
            {"role": "user", "content": chat_prompt}
        ]
    )

    answer = response.choices[0].message.content
    st.session_state.chat_history.append((question, answer))

for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
