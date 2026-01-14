import os
import json
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import PorterStemmer
import joblib

def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="latin-1")
    except Exception:
        df = pd.read_csv(path)
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    elif {"label", "text"}.issubset(df.columns):
        df = df[["label", "text"]]
    else:
        raise ValueError("Unsupported CSV format. Expected columns v1,v2 or label,text")
    df = df.dropna()
    return df

from utils import clean_text

def build_vectorizer():
    return TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english", preprocessor=clean_text)

def build_models(vectorizer: TfidfVectorizer):
    models = {}
    models["Naive Bayes"] = Pipeline([
        ("tfidf", vectorizer),
        ("clf", MultinomialNB()),
    ])
    models["Logistic Regression"] = Pipeline([
        ("tfidf", vectorizer),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    models["Support Vector Machine"] = Pipeline([
        ("tfidf", vectorizer),
        ("clf", CalibratedClassifierCV(LinearSVC(), cv=5)),
    ])
    models["Random Forest"] = Pipeline([
        ("tfidf", vectorizer),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
    ])
    models["K-Nearest Neighbors"] = Pipeline([
        ("tfidf", vectorizer),
        ("clf", KNeighborsClassifier(n_neighbors=5)),
    ])
    return models

def evaluate_models(models, X_train, X_test, y_train, y_test, y_train_bin, y_test_bin):
    results = {}
    pr_curves = {}
    roc_curves = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        # Test metrics
        y_pred_test = model.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        prec_test = precision_score(y_test, y_pred_test, pos_label="spam")
        rec_test = recall_score(y_test, y_pred_test, pos_label="spam")
        f1_test = f1_score(y_test, y_pred_test, pos_label="spam")
        if hasattr(model.named_steps["clf"], "predict_proba"):
            y_score_test = model.predict_proba(X_test)[:, 1]
        elif hasattr(model.named_steps["clf"], "decision_function"):
            y_score_test = model.decision_function(X_test)
        else:
            y_score_test = None
        auc_test = roc_auc_score(y_test_bin, y_score_test) if y_score_test is not None else float("nan")

        # Train metrics
        y_pred_train = model.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)
        prec_train = precision_score(y_train, y_pred_train, pos_label="spam")
        rec_train = recall_score(y_train, y_pred_train, pos_label="spam")
        f1_train = f1_score(y_train, y_pred_train, pos_label="spam")
        if hasattr(model.named_steps["clf"], "predict_proba"):
            y_score_train = model.predict_proba(X_train)[:, 1]
        elif hasattr(model.named_steps["clf"], "decision_function"):
            y_score_train = model.decision_function(X_train)
        else:
            y_score_train = None
        auc_train = roc_auc_score(y_train_bin, y_score_train) if y_score_train is not None else float("nan")

        results[name] = {
            "test": {
                "accuracy": acc_test,
                "precision": prec_test,
                "recall": rec_test,
                "f1_score": f1_test,
                "roc_auc": auc_test,
            },
            "train": {
                "accuracy": acc_train,
                "precision": prec_train,
                "recall": rec_train,
                "f1_score": f1_train,
                "roc_auc": auc_train,
            }
        }
        if y_score_test is not None:
            p, r, _ = precision_recall_curve(y_test_bin, y_score_test)
            fpr, tpr, _ = roc_curve(y_test_bin, y_score_test)
            pr_curves[name] = (r, p)
            roc_curves[name] = (fpr, tpr)
    return results, pr_curves, roc_curves

def save_visualizations(results, pr_curves, roc_curves, y_test, y_pred_best, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    names = list(results.keys())
    accuracies = [results[n]["test"]["accuracy"] for n in names]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=names, y=accuracies)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Test Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_accuracy_comparison.png"))
    plt.close()
    plt.figure(figsize=(8, 6))
    for name, (r, p) in pr_curves.items():
        plt.plot(r, p, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_recall_curves.png"))
    plt.close()
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr) in roc_curves.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"))
    plt.close()
    cm = confusion_matrix(y_test, y_pred_best, labels=["ham", "spam"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_best.png"))
    plt.close()

def train_and_save_best(input_path: str = "spam.csv", models_out_dir="models", artifacts_dir="artifacts"):
    os.makedirs(models_out_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    df = load_dataset(input_path)
    X = df["text"].astype(str)
    y = df["label"].astype(str)
    y_bin = (y == "spam").astype(int)
    X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
        X, y, y_bin, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = build_vectorizer()
    models = build_models(vectorizer)
    results, pr_curves, roc_curves = evaluate_models(models, X_train, X_test, y_train, y_test, y_train_bin, y_test_bin)
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    best_name = max(results.keys(), key=lambda n: results[n]["test"]["f1_score"])
    best_model = models[best_name]
    # Save all models
    for name, model in models.items():
        model_path = os.path.join(models_out_dir, f"{name.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, model_path)
    # Also save best as best_model for app compatibility
    best_path = os.path.join(models_out_dir, "best_model.joblib")
    joblib.dump(best_model, best_path)
    y_pred_best = best_model.predict(X_test)
    save_visualizations(results, pr_curves, roc_curves, y_test, y_pred_best, artifacts_dir)
    return best_path, results, best_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="spam.csv")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--artifacts_dir", default="artifacts")
    args = parser.parse_args()
    best_path, results, best_name = train_and_save_best(args.data, args.models_dir, args.artifacts_dir)
    print(json.dumps({"best_model": best_name, "saved_to": best_path, "metrics": results[best_name]}, indent=2))

if __name__ == "__main__":
    main()