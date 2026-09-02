Markdown
# 📱 Daraz AI Review Insight & Smartphone Sentiment NLP Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-orange?logo=huggingface)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![License](https://img.shields.io/badge/License-MIT-success)

## 📌 Project Overview
This repository contains an end-to-end Natural Language Processing (NLP) pipeline and a practical consumer application. It starts with interpreting public opinion from large volumes of smartphone reviews scraped from **GSMArena**, evaluating traditional ML algorithms against a fine-tuned **BERT** model. 

Taking these findings into the real world, this project also includes a **Custom Chrome Extension** designed for **Daraz.lk**. The extension connects to the fine-tuned BERT model via a FastAPI backend to automatically analyze Daraz product reviews, extracting aspect-based pros and cons in real-time to help users make informed purchasing decisions.

## ✨ Key Features
### Part 1: NLP Training Pipeline (Research & Model)
* **Custom Web Scraper:** Automated data collection extracting user reviews across multiple smartphone tiers.
* **Automated Annotation & Preprocessing:** Rule-based sentiment labeling utilizing NLTK's VADER Lexicon and robust text cleaning.
* **Deep Learning Fine-Tuning:** Custom `WeightedTrainer` implementation for `bert-base-uncased` to gracefully handle class imbalances, significantly outperforming traditional ML models (SVM, Random Forest).
* **Advanced Visualizations:** Generation of normalized confusion matrices and multi-dimensional radar plots.

### Part 2: Daraz Chrome Extension (Real-World Application)
* **Smart Sentiment Breakdown**: Automatically calculates Good 👍, Average 😐, and Bad 👎 percentages from product pages.
* **Auto Pros & Cons Extractor**: Uses Aspect-Based Sentiment Analysis to extract targeted product features (e.g., Battery, Camera) into structured columns.
* **Deep Scanning Algorithm**: Automatically handles page scrolling to bypass Daraz lazy-loading and fetch hidden reviews.
* **Daraz-Native UI**: A modern, interactive interface featuring SVG circular progress charts that perfectly blends with the Daraz theme.

---

## 🚀 How to Run Locally

To run the Chrome Extension on your machine, you need to download the pre-trained model, start the backend Python server, and then load the frontend extension into Google Chrome.

### 📥 Step 1: Download the Pre-trained BERT Model
Due to GitHub's file size limits, the fine-tuned BERT model is hosted externally on Google Drive.
1. **Download the Model:** [Click here to download bert_outputs.zip](https://drive.google.com/file/d/1PSfwaAHYRKfVqW05hdZN2ArQdmTbfR2a/view?usp=drive_link)
2. **Extract:** Extract the downloaded zip file and place the contents into the `mini NLP/bert_outputs/` directory in this repository.

### 🖥️ Step 2: Start the Backend Server (FastAPI)
1. Open your terminal and navigate to the `mini NLP` folder.
2. Install the required Python dependencies:
   ```bash
   pip install fastapi uvicorn pydantic transformers torch
Run the FastAPI server:

Bash
uvicorn api:app --reload
The backend should now be running on http://127.0.0.1:8000. Leave this terminal open.

### 🧩 Step 3: Load the Chrome Extension

Open Google Chrome and go to chrome://extensions/.

Turn on "Developer mode" (toggle switch in the top right corner).

Click the "Load unpacked" button in the top left.

Select the Extension folder from this repository.

The "Daraz AI Sentiment Analyzer" will now appear in your extensions list!


### 🛒 Step 4: Test the AI!

Go to any product page on Daraz.lk that has customer reviews.

The AI Review Insight widget will automatically appear below the product title.

Click "Deep Scan & Analyze Reviews" and watch the magic happen!

## 🗂️ Repository Structure

```text
├── mini NLP/                   # AI Model & Backend Server
│   ├── api.py                  # FastAPI server script
│   ├── data_scraper.py         # GSMArena robust scraping script
│   ├── preprocess_and_label.py # Text cleaning and VADER annotation
│   ├── baseline_models.py      # Traditional ML training pipeline
│   ├── bert_model.py           # Hugging Face BERT fine-tuning script
│   ├── advanced_comparison.py  # Evaluation visualization generator
│   └── bert_outputs/           # (Create this folder and place extracted model here)
│
├── Extension/                  # Chrome Extension Source Code
│   ├── manifest.json
│   ├── background.js
│   └── content.js
│
├── .gitignore
└── README.md
