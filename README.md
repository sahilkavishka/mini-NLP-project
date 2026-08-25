# 📱 Sentiment Analysis of Smartphone Reviews: Traditional ML vs. BERT

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-orange?logo=huggingface)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![License](https://img.shields.io/badge/License-MIT-success)

## 📌 Project Overview
This repository contains an end-to-end Natural Language Processing (NLP) pipeline designed to automatically interpret public opinion from large volumes of online textual data. Focusing on real-world mobile phone reviews scraped from **GSMArena**, this project comparatively evaluates the performance of traditional Machine Learning algorithms (Naïve Bayes, Logistic Regression, SVM, Random Forest) against a state-of-the-art Transformer-based architecture (**BERT**).

The findings demonstrate the profound superiority of transformer models in capturing deep semantic meaning, sequence, and context, bridging the gap between raw web data and actionable business intelligence.

## 🚀 Key Features
* **Custom Web Scraper:** Automated data collection pipeline extracting user reviews across multiple smartphone tiers (Flagship, Mid-range, Budget).
* **Automated Annotation:** Rule-based sentiment labeling utilizing NLTK's VADER Lexicon.
* **Robust Text Preprocessing:** Includes URL/Emoji removal, lemmatization, and custom contraction handling for varying model architectures.
* **Baseline ML Pipeline:** TF-IDF feature extraction chained with hyperparameter-tuned Scikit-Learn models.
* **Deep Learning Fine-Tuning:** Custom `WeightedTrainer` implementation for `bert-base-uncased` to gracefully handle class imbalances.
* **Advanced Visualizations:** Generation of normalized confusion matrices, grouped bar charts, and multi-dimensional radar plots.

## 🗂️ Repository Structure

```text
├── baseline_outputs/               # Saved Baseline models and TF-IDF vectorizers
├── bert_outputs/                   # Fine-tuned BERT model weights and tokenizer
├── final_evaluations/              # High-res comparison charts (Radar, Bar charts)
├── step1_data_scraper.py           # GSMArena robust scraping script
├── step2_preprocess_and_label.py   # Text cleaning and VADER annotation
├── step3_baseline_models.py        # TF-IDF & Traditional ML training pipeline
├── step4_bert_model.py             # Hugging Face BERT fine-tuning script
├── step5_advanced_comparison.py    # Evaluation visualization generator
└── README.md
