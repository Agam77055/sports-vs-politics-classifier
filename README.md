<div align="center">

# Sports vs Politics Text Classification

### Comparing Feature Representations and ML Classifiers for News Article Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**CSL4050 -- Natural Language Understanding | Assignment 1, Problem 4**

*Agam Harpreet Singh (B23CM1004)*

---

</div>

## Overview

This project builds a binary text classifier to distinguish **Sports** from **Politics** news articles. We systematically evaluate **9 combinations** of feature representations and machine learning classifiers, and report results on a held-out test set.

> **Key Result:** 7 out of 9 combinations achieve **perfect classification** (100% accuracy). All combinations exceed 98% accuracy.

---

## Dataset

| | Count | Percentage |
|---|---|---|
| **Sports** | 531 | 54.9% |
| **Politics** | 437 | 45.1% |
| **Total** | **968** | 100% |

**Sources:**
- [BBC News Dataset](http://mlg.ucd.ie/datasets/bbc.html) (511 sport + 417 politics articles)
- 20 manually curated articles per category covering diverse subtopics

<p align="center">
  <img src="plots/class_distribution.png" width="450"/>
</p>

---

## Method

### Feature Representations

| Method | Description |
|--------|-------------|
| **Bag of Words** | Raw word count vectors (`CountVectorizer`, 5000 features) |
| **TF-IDF** | Term frequency weighted by inverse document frequency (`TfidfVectorizer`, 5000 features) |
| **Bigram TF-IDF** | TF-IDF with unigrams + bigrams (`ngram_range=(1,2)`, 5000 features) |

### Classifiers

| Classifier | Type |
|-----------|------|
| **Multinomial Naive Bayes** | Probabilistic (generative) |
| **Logistic Regression** | Linear (discriminative) |
| **Linear SVM** | Maximum-margin (discriminative) |

### Preprocessing Pipeline

```
Raw Text -> Lowercase -> Remove punctuation/numbers -> Remove stopwords -> Remove short tokens
```

---

## Results

### Performance Table

| Feature | Classifier | Accuracy | Precision | Recall | F1-Score |
|:--------|:-----------|:--------:|:---------:|:------:|:--------:|
| BoW | Naive Bayes | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| BoW | Logistic Regression | 0.9897 | 0.9906 | 0.9906 | 0.9906 |
| BoW | SVM | 0.9845 | 0.9813 | 0.9906 | 0.9859 |
| TF-IDF | Naive Bayes | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| TF-IDF | Logistic Regression | 0.9948 | 0.9907 | 1.0000 | 0.9953 |
| TF-IDF | SVM | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| Bigram TF-IDF | Naive Bayes | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| Bigram TF-IDF | Logistic Regression | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| Bigram TF-IDF | SVM | **1.0000** | 1.0000 | 1.0000 | 1.0000 |

### F1-Score Comparison

<p align="center">
  <img src="plots/f1_comparison.png" width="700"/>
</p>

### Word Clouds

<p align="center">
  <img src="plots/wordcloud_sports.png" width="420"/>
  <img src="plots/wordcloud_politics.png" width="420"/>
</p>

### Confusion Matrices

**Bag of Words**

<p align="center">
  <img src="plots/cm_BoW_Naive_Bayes.png" width="270"/>
  <img src="plots/cm_BoW_Logistic_Regression.png" width="270"/>
  <img src="plots/cm_BoW_SVM.png" width="270"/>
</p>

**TF-IDF**

<p align="center">
  <img src="plots/cm_TFIDF_Naive_Bayes.png" width="270"/>
  <img src="plots/cm_TFIDF_Logistic_Regression.png" width="270"/>
  <img src="plots/cm_TFIDF_SVM.png" width="270"/>
</p>

**Bigram TF-IDF**

<p align="center">
  <img src="plots/cm_Bigram_TFIDF_Naive_Bayes.png" width="270"/>
  <img src="plots/cm_Bigram_TFIDF_Logistic_Regression.png" width="270"/>
  <img src="plots/cm_Bigram_TFIDF_SVM.png" width="270"/>
</p>

---

## Key Findings

- **Bigram TF-IDF** is the strongest feature set -- perfect with all 3 classifiers.
- **Naive Bayes** is the most consistent classifier -- perfect with all 3 feature types.
- **TF-IDF weighting** helps over raw BoW by reducing the impact of common cross-class words like "said", "year", "new".
- The only errors (2-3 misclassifications) occur with **BoW + LR** and **BoW + SVM**, where raw counts give too much weight to non-discriminative terms.

---

## Quick Start

```bash
# clone the repo
git clone https://github.com/Agam77055/sports-vs-politics-classifier.git
cd sports-vs-politics-classifier

# install dependencies
pip install -r requirements.txt

# run the classifier
python B23CM1004_prob4.py
```

The script will train all 9 models, print results, and save plots to `plots/`.

---

## Project Structure

```
.
├── B23CM1004_prob4.py        # main classification script
├── collect_data.py           # data collection and assembly
├── report.tex                # detailed LaTeX report
├── requirements.txt          # Python dependencies
├── data/
│   └── dataset.csv           # 968 labelled articles
├── plots/
│   ├── class_distribution.png
│   ├── f1_comparison.png
│   ├── wordcloud_sports.png
│   ├── wordcloud_politics.png
│   └── cm_*.png              # 9 confusion matrix plots
└── results/
    └── results.csv           # metrics for all 9 experiments
```

---

## References

1. D. Greene and P. Cunningham, "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering," *Proc. ICML*, 2006.
2. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825-2830, 2011.
3. C. Manning, P. Raghavan, and H. Schutze, *Introduction to Information Retrieval*, Cambridge University Press, 2008.

---

<div align="center">
  <sub>CSL4050 Natural Language Understanding | IIT Mandi | February 2026</sub>
</div>
