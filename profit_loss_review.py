import streamlit as st
import pandas as pd
from openai import OpenAI
from gtts import gTTS
import os

# ---- CONFIGURATION ----
st.set_page_config(page_title="📊 Monthly P&L Financial Review", layout="wide")

# Set OpenAI key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

# ---- PAGE HEADER ----
st.title("📊 Monthly P&L Financial Review")
st.markdown("Upload your Excel-based Profit & Loss statement. A CFO-style report will be generated along with an audio summary.")

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader("📁 Upload P&L Excel File", type=["xlsx"])
summary_text = ""

# ---- SUMMARY GENERATION FUNCTION ----
def generate_summary(df):
    df_clean = df.dropna(subset=["Actual", "Budget"]).copy()
    df_clean["Variance"] = df_clean["Actual"] - df_clean["Budget"]
    df_clean["Variance %"] = df_clean["Variance"] / df_clean["Budget"] * 100

    key_items = df_clean[abs(df_clean["Variance %"]) > 10].copy()
    key_items_str = key_items.to_string(index=False)

    prompt = f"""
You are a CFO summarizing monthly financials to a CEO. Below is the P&L summary showing actuals vs. budget with variance:

{key_items_str}

Create a 3-5 minute audio-style summary in plain English. Highlight revenue growth, cost overruns, savings, cash flow trends, and any major areas to watch. Use a confident, conversational tone as if you're presenting during a board meeting.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# ---- MAIN LOGIC ----
if uploaded_file:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    st.subheader("📄 Preview of Uploaded File")
    st.dataframe(df)

    if st.button("📊 Generate CFO Summary"):
        summary_text = generate_summary(df)
        st.session_state["summary_text"] = summary_text
        st.text_area("📝 CFO Summary Transcript", summary_text, height=300)

# ---- AUDIO PLAYBACK ----
if "summary_text" in st.session_state:
    if st.button("🔊 Listen to Summary"):
        tts = gTTS(st.session_state["summary_text"])
        audio_file = "summary_audio.mp3"
        tts.save(audio_file)
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")
        os.remove(audio_file)

# ---- OPTIONAL CHAT-BASED Q&A ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("💬 Ask Questions About This Month's Financials")
question = st.text_input("Type your question and press Enter")

if st.button("Ask") and question:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a CFO answering questions based on the company's P&L."},
            {"role": "user", "content": question}
        ]
    )
    answer = response.choices[0].message.content
    st.session_state.chat_history.append((question, answer))

for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
