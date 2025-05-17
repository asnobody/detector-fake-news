# app.py
import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

@st.cache_data
def load_data():
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")

    df_fake["class"] = 0
    df_true["class"] = 1

    df = pd.concat([df_fake, df_true])
    df = df.drop(columns=["title", "subject", "date"])
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@st.cache_resource
def train_models(df):
    df["text"] = df["text"].apply(preprocess_text)
    X = df["text"]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(random_state=0)
    }

    for name in models:
        models[name].fit(X_train_vec, y_train)

    return models, vectorizer

def get_label(n):
    return "Fake News" if n == 0 else "Not a Fake News"

# Streamlit App
st.title("📰 Detector de Fake News")
st.write("Insira uma notícia abaixo para verificar se é verdadeira ou falsa.")

news_input = st.text_area("Digite a notícia aqui:")

if st.button("Verificar"):
    if news_input.strip() == "":
        st.warning("Por favor, insira uma notícia.")
    else:
        df = load_data()
        models, vectorizer = train_models(df)
        cleaned = preprocess_text(news_input)
        vectorized = vectorizer.transform([cleaned])

        st.subheader("Resultados das Previsões:")
        for name, model in models.items():
            prediction = model.predict(vectorized)[0]
            st.write(f"**{name}**: {get_label(prediction)}")
