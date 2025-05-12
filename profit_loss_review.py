import streamlit as st
import pandas as pd
import openai
from io import BytesIO
from datetime import datetime

# ---- Config ----
st.set_page_config(page_title="Monthly Financial Summary", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---- UI ----
st.title("Monthly P&L Review")
st.markdown("Upload your Excel-based Profit & Loss statement for a CFO-style financial summary.")

uploaded_file = st.file_uploader("Upload P&L Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Preview of Uploaded File")
    st.dataframe(df.head())

    # ---- Prepare Input for GPT ----
    def generate_summary(df):
        # Basic assumption: P&L has columns: Category | Actual | Budget
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

    if st.button("Generate CFO Summary"):
        summary_text = generate_summary(df)
        st.subheader("CFO-Style Audio Summary")
        st.text_area("Transcript", summary_text, height=300)

        # Optionally generate TTS with third-party service
        st.markdown("*You can now copy this transcript into a voice generation tool like ElevenLabs, Play.ht, or Amazon Polly to create your audio snippet.*")

# ---- Optional: Add Q&A Interface ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("Ask Questions About This Month's Financials")

question = st.text_input("Type your question:")
if st.button("Ask") and question:
    chat_prompt = f"You are a CFO reviewing the company's P&L. Here's the user's question: {question}"
    chat_history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])

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
