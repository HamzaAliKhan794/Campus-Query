import csv
import time
import re
from datetime import datetime

from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

# CLEANING FUNCTION
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

# LOAD DATASET
faq = pd.read_csv("faq.csv").fillna("")
faq["clean_question"] = faq["question"].apply(clean)


# TF-IDF VECTOR SETUP
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq["clean_question"])


# LOGGING UNKNOWN QUESTIONS
def log_unknown_question(question):
    with open("unknown_questions.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            question,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])


# ANSWER FUNCTION
def get_answer(question):
    clean_q = clean(question)
    user_vec = vectorizer.transform([clean_q])

    sims = cosine_similarity(user_vec, faq_vectors)[0]
    best_score = sims.max()
    best_index = sims.argmax()

    # Fallback
    if best_score < 0.35:
        log_unknown_question(question)
        return (
            "Sorry, I don’t have this information. "
            "Please contact the IILM helpdesk at +91-8860602497"
        )

    return faq.iloc[best_index]["answer"]


# ROUTES
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    # 1.5s delay
    time.sleep(1.5)

    answer = get_answer(question)
    return jsonify({"answer": answer})


# RUN SERVER
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

