
import os
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from openai import OpenAI
import streamlit as st

# ---------- 1. DATA (using Pandas) ----------
CSV_PATH = "sentiment.csv"
df = pd.read_csv(CSV_PATH)

# ---------- 2. TEXT PRE-PROCESSING (spaCy) ----------
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # speed boost

def clean(text: str) -> str:
    doc = nlp(text.lower())
    tokens = [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]
    return " ".join(tokens)

df["clean"] = df["text"].apply(clean)

# ---------- 3. TRAIN MODEL (scikit-learn) ----------
X_train, X_test, y_train, y_test = train_test_split(
    
    df["clean"], df["label"], test_size=0.2, random_state=42
)
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
baseline_acc = model.score(X_test, y_test)

# ---------- 4. OPENAI GPT HELPER ----------
client = OpenAI(api_key="your-real-openai-api-key")
def gpt_sentiment(sentence: str) -> str:
    """Return 'positive' or 'negative' according to GPT-4"""
    prompt = (
        "Classify the sentiment (positive/negative ONLY) of the following sentence:\n"
        f"{sentence}\nAnswer:"
    )
    resp = client.Chat.Completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip().lower()

# ---------- 5. STREAMLIT UI ----------
st.title("🤖 Sentiment Analysis 🤖")


sentence = st.text_input("Type a sentence and hit Enter:")

if sentence:
    # ML prediction
    ml_pred = model.predict([clean(sentence)])[0]


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("      Final Prediction 🎯 ")
        st.write(f"**➡️{ml_pred.capitalize()}**")


   
