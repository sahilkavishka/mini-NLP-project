import os
import re
import string
import pandas as pd
import numpy as np
from tqdm import tqdm

import nltk
import contractions
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# 1. Download Required NLTK Resources safely
nltk_resources = ['vader_lexicon', 'stopwords', 'wordnet', 'punkt', 'punkt_tab']
for res in nltk_resources:
    nltk.download(res, quiet=True)

# 2. Text Preprocessing Classes & Functions
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        # Sentiment එකට වැදගත් වන negation words stopwords වලින් ඉවත් කර තබා ගැනීම
        self.stop_words = set(stopwords.words('english')) - {
            'not', 'no', 'nor', 'neither', 'never', 'hardly', 'barely', 'scarcely', 'against', 'but'
        }

    def clean_for_bert(self, text: str) -> str:
        """
        BERT සඳහා text එක සකස් කිරීම:
        HTML, URLs, අනවශ්‍ය Special characters ඉවත් කරයි. 
        නමුත් Capitalization, Punctuations සහ Sentence structure එලෙසම තබා ගනී.
        """
        if not isinstance(text, str):
            return ""
        
        # HTML tags ඉවත් කිරීම
        text = re.sub(r'<.*?>', '', text)
        # URLs ඉවත් කිරීම
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Non-ASCII emojis / strange symbols ඉවත් කිරීම
        text = text.encode('ascii', 'ignore').decode('ascii')
        # Contractions expand කිරීම (e.g., "don't" -> "do not")
        text = contractions.fix(text)
        # Multiple spaces / newlines ඉවත් කිරීම
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def clean_for_baseline(self, bert_text: str) -> str:
        """
        TF-IDF සහ Traditional ML Models (Naïve Bayes, SVM) සඳහා text එක සකස් කිරීම:
        Lowercasing, Punctuation ඉවත් කිරීම, Lemmatization සහ Stopwords ඉවත් කරයි.
        """
        if not bert_text:
            return ""
        
        # Lowercase
        text = bert_text.lower()
        # Remove Punctuations & Numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        tokens = word_tokenize(text)
        # Lemmatize and remove stopwords
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]
        
        return " ".join(cleaned_tokens)

# 3. VADER-based Rule-based Sentiment Annotation
def assign_sentiment(text: str, analyzer: SentimentIntensityAnalyzer):
    score = analyzer.polarity_scores(text)['compound']
    
    # Standard NLP VADER Compound Thresholds
    if score >= 0.05:
        label = 'Positive'
    elif score <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
        
    return label, score

def main():
    input_file = 'gsmarena_professional_dataset.csv'
    output_file = 'gsmarena_processed_dataset.csv'

    if not os.path.exists(input_file):
        print(f"[Error] '{input_file}' සොයාගත නොහැක. පළමුව Step 1 run කරන්න.")
        return

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    initial_count = len(df)
    print(f"Total raw records loaded: {initial_count}")

    # Remove Missing & Exact Duplicates
    df.dropna(subset=['review_text'], inplace=True)
    df.drop_duplicates(subset=['review_text'], inplace=True)

    preprocessor = TextPreprocessor()
    analyzer = SentimentIntensityAnalyzer()

    # Step A: Clean for BERT
    print("\n[1/4] Preparing natural text for BERT...")
    df['text_for_bert'] = [preprocessor.clean_for_bert(t) for t in tqdm(df['review_text'])]

    # Filter out noisy/very short reviews (< 3 words)
    df['word_count'] = df['text_for_bert'].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= 3].reset_index(drop=True)

    # Step B: Automated Sentiment Labeling
    print("\n[2/4] Annotating Sentiment Labels using VADER...")
    sentiments = []
    compound_scores = []
    for text in tqdm(df['text_for_bert']):
        label, score = assign_sentiment(text, analyzer)
        sentiments.append(label)
        compound_scores.append(score)

    df['sentiment'] = sentiments
    df['sentiment_score'] = compound_scores

    # Step C: Clean for Baseline Models (TF-IDF)
    print("\n[3/4] Generating Lemmatized Text for Baseline Models...")
    df['text_for_baseline'] = [preprocessor.clean_for_baseline(t) for t in tqdm(df['text_for_bert'])]

    # Drop any entries that became empty after baseline cleaning
    df = df[df['text_for_baseline'].str.strip().astype(bool)].reset_index(drop=True)

    # Map labels to integers for ML Models (0: Negative, 1: Neutral, 2: Positive)
    label_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['label'] = df['sentiment'].map(label_mapping)

    # Step D: Stratified Train / Validation / Test Split (80% / 10% / 10%)
    print("\n[4/4] Creating Stratified Splits (80% Train, 10% Val, 10% Test)...")
    # 80% Train, 20% Temp
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['label']
    )
    # Split the 20% temp equally into 10% Val and 10% Test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['label']
    )

    # Tag splits inside dataframe
    df['split'] = 'train'
    df.loc[val_df.index, 'split'] = 'validation'
    df.loc[test_df.index, 'split'] = 'test'

    # Save to disk
    df.to_csv(output_file, index=False, encoding='utf-8')
    train_df.to_csv('train.csv', index=False, encoding='utf-8')
    val_df.to_csv('val.csv', index=False, encoding='utf-8')
    test_df.to_csv('test.csv', index=False, encoding='utf-8')

    print(f"\nSuccessfully created processed dataset: '{output_file}'")
    print(f"Splits saved: train.csv ({len(train_df)}), val.csv ({len(val_df)}), test.csv ({len(test_df)})")

    # Dataset Summary & Class Distribution
    print("\n=== Dataset Distribution Summary ===")
    summary = df.groupby(['split', 'sentiment']).size().unstack(fill_value=0)
    print(summary)
    print("\nData Sample:")
    print(df[['text_for_bert', 'text_for_baseline', 'sentiment', 'split']].head(3))

if __name__ == "__main__":
    main()