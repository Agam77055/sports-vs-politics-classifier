"""
B23CM1004_prob4.py — Sports vs Politics Text Classifier
========================================================
Reads data/dataset.csv containing sports and politics articles,
preprocesses the text, extracts features using three different
representations (BoW, TF-IDF, Bigram TF-IDF), trains three
different ML classifiers (Naive Bayes, Logistic Regression, SVM),
and evaluates all 9 combinations.

Generates:
  - Comparison table (printed + saved as CSV)
  - Confusion matrix heatmaps
  - F1-score bar chart
  - Word clouds for each class
  - Class distribution chart

Usage:
    python B23CM1004_prob4.py
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# make sure output directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# random seed for reproducibility
SEED = 42

# ============================================================
# PREPROCESSING
# ============================================================

# common English stopwords (hardcoded to avoid NLTK dependency)
STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "can", "need", "must", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "our", "their",
    "not", "no", "nor", "so", "as", "if", "then", "than", "too",
    "very", "just", "about", "above", "after", "again", "all",
    "also", "am", "any", "because", "before", "below", "between",
    "both", "each", "few", "further", "get", "got", "here", "how",
    "into", "more", "most", "new", "now", "only", "other", "out",
    "over", "own", "same", "some", "such", "there", "through",
    "under", "until", "up", "what", "when", "where", "which",
    "while", "who", "whom", "why", "down", "during", "s", "t",
    "don", "didn", "doesn", "hadn", "hasn", "haven", "isn",
    "wasn", "weren", "won", "wouldn", "said", "also", "one", "two"
])


def preprocess_text(text):
    """
    Clean and preprocess a single text string.
    Steps:
      1. Lowercase everything
      2. Remove punctuation and numbers
      3. Remove extra whitespace
      4. Remove stopwords
    """
    # lowercase
    text = text.lower()
    # remove punctuation and numbers — keep only letters and spaces
    text = re.sub(r"[^a-z\s]", "", text)
    # split into words, remove stopwords, rejoin
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def get_vectorizers():
    """
    Return a dictionary of (name, vectorizer) pairs.
    We compare three feature representations:
      1. Bag of Words — raw word counts
      2. TF-IDF — term frequency * inverse document frequency
      3. Bigram TF-IDF — TF-IDF with unigrams + bigrams
    """
    return {
        "BoW": CountVectorizer(max_features=5000),
        "TF-IDF": TfidfVectorizer(max_features=5000),
        "Bigram TF-IDF": TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
    }


# ============================================================
# CLASSIFIERS
# ============================================================

def get_classifiers():
    """
    Return a dictionary of (name, classifier) pairs.
    We compare three ML techniques:
      1. Multinomial Naive Bayes
      2. Logistic Regression
      3. Linear SVM (Support Vector Machine)
    """
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "SVM": LinearSVC(max_iter=2000, random_state=SEED),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(y_true, y_pred, model_name, feature_name):
    """
    Compute and return evaluation metrics for one experiment.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="sports", average="binary")
    rec = recall_score(y_true, y_pred, pos_label="sports", average="binary")
    f1 = f1_score(y_true, y_pred, pos_label="sports", average="binary")
    cm = confusion_matrix(y_true, y_pred, labels=["sports", "politics"])

    return {
        "Feature": feature_name,
        "Classifier": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Confusion Matrix": cm,
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_confusion_matrix(cm, title, filename):
    """Save a confusion matrix heatmap as an image."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sports", "Politics"],
                yticklabels=["Sports", "Politics"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()


def plot_f1_comparison(results_df):
    """Bar chart comparing F1-scores across all 9 experiments."""
    fig, ax = plt.subplots(figsize=(10, 5))
    # create a combined label for each experiment
    labels = results_df["Feature"] + "\n+ " + results_df["Classifier"]
    colors = []
    palette = {"Naive Bayes": "#4C72B0", "Logistic Regression": "#55A868", "SVM": "#C44E52"}
    for clf in results_df["Classifier"]:
        colors.append(palette.get(clf, "#999999"))

    bars = ax.bar(range(len(labels)), results_df["F1-Score"], color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score Comparison: Features x Classifiers")
    ax.set_ylim(0, 1.05)

    # add value labels on bars
    for bar, val in zip(bars, results_df["F1-Score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n) for n, c in palette.items()]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "f1_comparison.png"), dpi=150)
    plt.close()


def plot_class_distribution(df):
    """Bar chart showing class distribution in the dataset."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts()
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title("Class Distribution in Dataset")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Articles")
    ax.set_xticklabels(counts.index, rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150)
    plt.close()


def plot_wordclouds(df):
    """Generate word clouds for each class."""
    for label in ["sports", "politics"]:
        text = " ".join(df[df["label"] == label]["clean_text"])
        wc = WordCloud(width=800, height=400, background_color="white",
                       max_words=100, colormap="viridis").generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Word Cloud — {label.capitalize()}", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"wordcloud_{label}.png"), dpi=150)
        plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  Sports vs Politics Text Classifier")
    print("=" * 60)

    # --- Load data ---
    print("\n[1] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"    Total articles: {len(df)}")
    print(f"    Sports: {len(df[df['label'] == 'sports'])}")
    print(f"    Politics: {len(df[df['label'] == 'politics'])}")

    # --- Preprocess ---
    print("\n[2] Preprocessing text...")
    df["clean_text"] = df["text"].apply(preprocess_text)
    # drop any rows that ended up empty after cleaning
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"    Articles after cleaning: {len(df)}")

    # --- Plot class distribution and word clouds ---
    print("\n[3] Generating visualizations...")
    plot_class_distribution(df)
    plot_wordclouds(df)
    print("    Saved class_distribution.png, wordcloud_sports.png, wordcloud_politics.png")

    # --- Train/test split ---
    print("\n[4] Splitting data (80% train / 20% test, stratified)...")
    X_text = df["clean_text"]
    y = df["label"]
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"    Train: {len(X_train_text)} | Test: {len(X_test_text)}")

    # --- Run all 9 experiments (3 features x 3 classifiers) ---
    print("\n[5] Running experiments (3 features x 3 classifiers = 9 combos)...\n")
    vectorizers = get_vectorizers()
    classifiers = get_classifiers()
    all_results = []

    for feat_name, vectorizer in vectorizers.items():
        # fit vectorizer on training data, transform both
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        for clf_name, clf in classifiers.items():
            # train the classifier
            clf.fit(X_train, y_train)
            # predict on test set
            y_pred = clf.predict(X_test)

            # evaluate
            result = evaluate_model(y_test, y_pred, clf_name, feat_name)
            all_results.append(result)

            # print a quick summary
            print(f"    {feat_name:15s} + {clf_name:22s} | "
                  f"Acc: {result['Accuracy']:.4f} | F1: {result['F1-Score']:.4f}")

            # save confusion matrix plot
            cm_filename = f"cm_{feat_name.replace(' ', '_').replace('-', '')}_{clf_name.replace(' ', '_')}.png"
            plot_confusion_matrix(
                result["Confusion Matrix"],
                f"{feat_name} + {clf_name}",
                cm_filename
            )

        # re-create classifiers for the next feature rep (fresh instances)
        classifiers = get_classifiers()

    # --- Build results table ---
    print("\n[6] Results summary:\n")
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "Confusion Matrix"}
        for r in all_results
    ])
    print(results_df.to_string(index=False))

    # save to CSV
    results_csv = os.path.join(RESULTS_DIR, "results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n    Results saved to {results_csv}")

    # --- F1 comparison bar chart ---
    plot_f1_comparison(results_df)
    print("    Saved f1_comparison.png")

    # --- Find best combination ---
    best_idx = results_df["F1-Score"].idxmax()
    best = results_df.iloc[best_idx]
    print(f"\n    Best combination: {best['Feature']} + {best['Classifier']}")
    print(f"    Accuracy: {best['Accuracy']:.4f} | F1: {best['F1-Score']:.4f}")

    # --- Print detailed classification report for best model ---
    print(f"\n[7] Detailed report for best model ({best['Feature']} + {best['Classifier']}):\n")
    # re-train best model to get predictions
    best_vec = get_vectorizers()[best["Feature"]]
    best_clf = get_classifiers()[best["Classifier"]]
    X_tr = best_vec.fit_transform(X_train_text)
    X_te = best_vec.transform(X_test_text)
    best_clf.fit(X_tr, y_train)
    y_pred_best = best_clf.predict(X_te)
    print(classification_report(y_test, y_pred_best, target_names=["politics", "sports"]))

    print("\nDone! All plots saved to:", PLOTS_DIR)
    print("All results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
