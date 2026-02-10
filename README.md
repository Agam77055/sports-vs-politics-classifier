# Sports vs Politics Text Classifier

**CSL4050 - Natural Language Understanding | Assignment 1, Problem 4**

**Author:** Agam Harpreet Singh (B23CM1004)

## About

A text classification system that categorizes news articles as Sports or Politics. We compare 3 feature representations with 3 ML classifiers (9 combinations total) and evaluate them on accuracy, precision, recall, and F1-score.

## Dataset

968 articles total:
- 511 sports + 20 curated = 531 sports articles
- 417 politics + 20 curated = 437 politics articles

Source: [BBC News Dataset](http://mlg.ucd.ie/datasets/bbc.html) + manually written articles.

## Features and Classifiers

| Features | Classifiers |
|----------|------------|
| Bag of Words (BoW) | Multinomial Naive Bayes |
| TF-IDF | Logistic Regression |
| Bigram TF-IDF | Linear SVM |

## Results

| Feature | Classifier | Accuracy | F1-Score |
|---------|-----------|----------|----------|
| BoW | Naive Bayes | 1.0000 | 1.0000 |
| BoW | Logistic Regression | 0.9897 | 0.9906 |
| BoW | SVM | 0.9845 | 0.9859 |
| TF-IDF | Naive Bayes | 1.0000 | 1.0000 |
| TF-IDF | Logistic Regression | 0.9948 | 0.9953 |
| TF-IDF | SVM | 1.0000 | 1.0000 |
| Bigram TF-IDF | Naive Bayes | 1.0000 | 1.0000 |
| Bigram TF-IDF | Logistic Regression | 1.0000 | 1.0000 |
| Bigram TF-IDF | SVM | 1.0000 | 1.0000 |

7 out of 9 combinations achieve perfect classification. Bigram TF-IDF is perfect across all classifiers.

### F1-Score Comparison

![F1 Comparison](plots/f1_comparison.png)

### Word Clouds

| Sports | Politics |
|--------|----------|
| ![Sports](plots/wordcloud_sports.png) | ![Politics](plots/wordcloud_politics.png) |

### Sample Confusion Matrices

| BoW + Naive Bayes | BoW + Logistic Regression | BoW + SVM |
|---|---|---|
| ![](plots/cm_BoW_Naive_Bayes.png) | ![](plots/cm_BoW_Logistic_Regression.png) | ![](plots/cm_BoW_SVM.png) |

## How to Run

```bash
pip install -r requirements.txt
python B23CM1004_prob4.py
```

## Files

```
prob4/
├── B23CM1004_prob4.py      # main script
├── collect_data.py          # data collection script
├── report.tex               # detailed LaTeX report
├── requirements.txt
├── data/
│   └── dataset.csv          # the dataset
├── plots/                   # all generated plots
│   ├── class_distribution.png
│   ├── f1_comparison.png
│   ├── wordcloud_sports.png
│   ├── wordcloud_politics.png
│   └── cm_*.png             # confusion matrices (9 total)
└── results/
    └── results.csv          # results table
```
