"""
Fake News Detection - Core ML Engine
Provides text preprocessing, dataset loading, model training, evaluation,
model persistence (joblib), and multi-model inference.
"""

import os
import re
import sys
import time
import string
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)


def locate_data_file(filename):
    """Search for data files across common project paths."""
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "Datasets", filename),
        os.path.join(BASE_DIR, "Fake-News-Detection-main", filename),
        os.path.join(BASE_DIR, "Fake-News-Detection-main", "Datasets", filename),
        filename,
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {filename} in known paths: {candidates}")


def clean_text(text):
    """
    Cleans raw text by:
    - Lowercasing
    - Removing square brackets and their contents
    - Replacing non-word characters with spaces
    - Removing URLs
    - Removing HTML tags (fixing the Python 3 byte-string bug from the original notebook)
    - Removing punctuation
    - Removing alphanumeric tokens containing digits
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)  # fixed bug: was b'' in original notebook
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_dataset(sample_size=None, random_state=42):
    """
    Loads Fake.csv and True.csv, labels them (Fake=0, True=1),
    and combines them into a cleaned dataset.
    """
    fake_path = locate_data_file("Fake.csv")
    true_path = locate_data_file("True.csv")

    print(f"Loading datasets...\n  Fake: {fake_path}\n  True: {true_path}")
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    df_fake["class"] = 0
    df_true["class"] = 1

    if sample_size and sample_size < len(df_fake) and sample_size < len(df_true):
        half = sample_size // 2
        df_fake = df_fake.sample(n=half, random_state=random_state)
        df_true = df_true.sample(n=half, random_state=random_state)
        print(f"Sampled balanced subset of {half * 2} articles ({half} Fake, {half} True).")
    else:
        print(f"Using full dataset: {len(df_fake)} Fake, {len(df_true)} True (Total: {len(df_fake) + len(df_true)})")

    df = pd.concat([df_fake, df_true], axis=0, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("Cleaning text data...")
    t0 = time.time()
    df["clean_text"] = df["text"].apply(clean_text)
    print(f"Text cleaning completed in {time.time() - t0:.2f}s.")

    return df


class FakeNewsModelSuite:
    """Encapsulates vectorizer and all 4 classifiers."""

    MODEL_KEYS = ["LR", "DT", "RF", "GB"]
    MODEL_NAMES = {
        "LR": "Logistic Regression",
        "DT": "Decision Tree",
        "RF": "Random Forest",
        "GB": "Gradient Boosting"
    }

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.vectorizer = None
        self.models = {}
        self.metrics = {}

    def get_model_paths(self):
        return {
            "vec": os.path.join(self.models_dir, "vectorizer.joblib"),
            "LR": os.path.join(self.models_dir, "model_lr.joblib"),
            "DT": os.path.join(self.models_dir, "model_dt.joblib"),
            "RF": os.path.join(self.models_dir, "model_rf.joblib"),
            "GB": os.path.join(self.models_dir, "model_gb.joblib"),
            "metrics": os.path.join(self.models_dir, "metrics.joblib"),
        }

    def is_trained(self):
        paths = self.get_model_paths()
        return all(os.path.exists(p) for p in [paths["vec"], paths["LR"], paths["DT"]])

    def save(self):
        os.makedirs(self.models_dir, exist_ok=True)
        paths = self.get_model_paths()
        joblib.dump(self.vectorizer, paths["vec"])
        for k, model in self.models.items():
            if k in paths:
                joblib.dump(model, paths[k])
        if self.metrics:
            joblib.dump(self.metrics, paths["metrics"])
        print(f"Models successfully saved to: {self.models_dir}")

    def load(self):
        paths = self.get_model_paths()
        if not os.path.exists(paths["vec"]):
            raise FileNotFoundError(f"Vectorizer not found at {paths['vec']}. Please train models first.")
        self.vectorizer = joblib.load(paths["vec"])
        for k in self.MODEL_KEYS:
            if os.path.exists(paths[k]):
                self.models[k] = joblib.load(paths[k])
        if os.path.exists(paths["metrics"]):
            self.metrics = joblib.load(paths["metrics"])
        print(f"Loaded {len(self.models)} models and vectorizer from {self.models_dir}")
        return True

    def train(self, df, test_size=0.25, max_features=10000, random_state=42, skip_gb=False):
        """Train vectorizer and all 4 classifiers."""
        x = df["clean_text"]
        y = df["class"]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\nFitting TF-IDF Vectorizer (max_features={max_features})...")
        t0 = time.time()
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        xv_train = self.vectorizer.fit_transform(x_train)
        xv_test = self.vectorizer.transform(x_test)
        print(f"Vectorization completed in {time.time() - t0:.2f}s. Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        model_configs = [
            ("LR", LogisticRegression(max_iter=1000, random_state=random_state)),
            ("DT", DecisionTreeClassifier(random_state=random_state)),
            ("RF", RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=random_state)),
        ]

        if not skip_gb:
            # GradientBoostingClassifier with 40 estimators for fast convergence
            model_configs.append(("GB", GradientBoostingClassifier(n_estimators=40, random_state=random_state)))

        self.metrics = {}
        for key, model in model_configs:
            name = self.MODEL_NAMES[key]
            print(f"\n--- Training {name} ({key}) ---")
            t_start = time.time()
            model.fit(xv_train, y_train)
            train_time = time.time() - t_start

            y_pred = model.predict(xv_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=["Fake", "True"], output_dict=True)

            self.models[key] = model
            self.metrics[key] = {
                "name": name,
                "accuracy": round(acc, 4),
                "train_time_sec": round(train_time, 2),
                "report": report
            }
            print(f"Done in {train_time:.2f}s | Test Accuracy: {acc * 100:.2f}%")

        self.save()
        return self.metrics

    def predict_article(self, text):
        """
        Runs text through all available models and returns structured predictions.
        """
        if self.vectorizer is None or not self.models:
            self.load()

        cleaned = clean_text(text)
        vec_text = self.vectorizer.transform([cleaned])

        results = {}
        fake_votes = 0
        true_votes = 0

        for key, model in self.models.items():
            pred = int(model.predict(vec_text)[0])
            prob = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vec_text)[0]
                prob = {"fake": float(probs[0]), "true": float(probs[1])}

            label = "Not A Fake News" if pred == 1 else "Fake News"
            if pred == 1:
                true_votes += 1
            else:
                fake_votes += 1

            results[key] = {
                "name": self.MODEL_NAMES.get(key, key),
                "prediction": pred,
                "label": label,
                "probabilities": prob
            }

        consensus_label = "Not A Fake News" if true_votes > fake_votes else "Fake News"
        confidence = max(true_votes, fake_votes) / len(self.models)

        return {
            "cleaned_text_preview": cleaned[:120] + "..." if len(cleaned) > 120 else cleaned,
            "consensus": consensus_label,
            "consensus_class": 1 if consensus_label == "Not A Fake News" else 0,
            "confidence_ratio": f"{max(true_votes, fake_votes)}/{len(self.models)}",
            "confidence_percent": round(confidence * 100, 1),
            "votes": {"fake": fake_votes, "true": true_votes},
            "models": results
        }
