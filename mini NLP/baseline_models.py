import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # Required to enable HalvingGridSearchCV
from sklearn.model_selection import HalvingGridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# 1. Professional Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 2. Directory Setup
OUTPUT_DIR = "baseline_outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    """Load train, validation, and test datasets."""
    logger.info("Loading datasets...")
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')
    test_df = pd.read_csv('test.csv')
    
    # Fill missing with empty string
    for df in [train_df, val_df, test_df]:
        df['text_for_baseline'] = df['text_for_baseline'].fillna('')
        
    return train_df, val_df, test_df

def create_and_tune_pipeline(model_name, classifier, param_grid, X_train, y_train):
    """
    Creates a scikit-learn Pipeline and uses HalvingGridSearchCV 
    for fast, state-of-the-art hyperparameter tuning.
    """
    # Create Pipeline binding TF-IDF and the Model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            max_df=0.9
        )),
        ('clf', classifier)
    ])
    
    # Modern HalvingGridSearchCV - FIXED RESOURCE ERROR
    search = HalvingGridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        factor=2,
        resource='n_samples', # Strictly use n_samples when max_resources='auto'
        max_resources='auto',
        scoring='f1_weighted',
        random_state=42,
        n_jobs=-1
    )
    
    logger.info(f"Starting Hyperparameter Tuning for {model_name}...")
    search.fit(X_train, y_train)
    
    logger.info(f"Best parameters for {model_name}: {search.best_params_}")
    logger.info(f"Best Cross-Validation F1-Score: {search.best_score_:.4f}")
    
    return search.best_estimator_

def evaluate_and_plot(name, model, X_test, y_test, class_names):
    """Evaluate model and return metrics + confusion matrix."""
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    metrics = {
        'Model': name,
        'Accuracy': acc,
        'F1-Score (Weighted)': f1_w,
        'Precision (Weighted)': precision_w,
        'Recall (Weighted)': recall_w
    }
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    
    return metrics, report, cm

def plot_matrices(cms, class_names):
    """Generate high-quality confusion matrix visual."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, (name, cm) in enumerate(cms.items()):
        # Normalize confusion matrix for better interpretation
        # Added small epsilon to prevent division by zero in empty rows
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1
        cm_normalized = cm.astype('float') / row_sums
        
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2f', cmap='magma', ax=axes[idx],
            xticklabels=class_names, yticklabels=class_names,
            vmin=0, vmax=1
        )
        axes[idx].set_title(f'{name} (Normalized)', fontsize=14, pad=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=12)
        axes[idx].set_ylabel('True Label', fontsize=12)
        
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'baseline_confusion_matrices_pro.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"High-res Confusion matrices saved to: {plot_path}")

def main():
    train_df, val_df, test_df = load_data()
    class_names = ['Negative', 'Neutral', 'Positive']
    
    X_train, y_train = train_df['text_for_baseline'], train_df['label']
    X_test, y_test = test_df['text_for_baseline'], test_df['label']
    
    # Definitions of models with balanced class weights
    # Slide 12 models: Naïve Bayes, Logistic Regression, SVM, Random Forest[cite: 1]
    models_config = {
        'Naïve Bayes': {
            'clf': MultinomialNB(),
            'params': {
                'clf__alpha': [0.1, 0.5, 1.0, 2.0],
                'tfidf__max_features': [3000, 5000, 10000]
            }
        },
        'Logistic Regression': {
            'clf': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
            'params': {
                'clf__C': [0.1, 1.0, 10.0],
                'tfidf__max_features': [5000, 10000]
            }
        },
        'Support Vector Machine (SVM)': {
            'clf': LinearSVC(class_weight='balanced', random_state=42, dual=False),
            'params': {
                'clf__C': [0.01, 0.1, 1.0, 10.0],
                'tfidf__max_features': [5000, 10000]
            }
        },
        'Random Forest': {
            'clf': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
            'params': {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [None, 20, 50],
                'tfidf__max_features': [3000, 5000]
            }
        }
    }
    
    results_list = []
    confusion_matrices = {}
    best_pipelines = {}
    
    logger.info("=== Starting Advanced Model Training Pipeline ===")
    
    for name, config in models_config.items():
        logger.info(f"--- Processing {name} ---")
        
        # Train and Tune
        best_model = create_and_tune_pipeline(
            name, config['clf'], config['params'], X_train, y_train
        )
        best_pipelines[name] = best_model
        
        # Save best model
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("ï", "i")
        joblib.dump(best_model, os.path.join(MODEL_DIR, f"{safe_name}_pipeline.pkl"))
        
        # Evaluate on Test Set
        metrics, report, cm = evaluate_and_plot(name, best_model, X_test, y_test, class_names)
        results_list.append(metrics)
        confusion_matrices[name] = cm
        
        print(f"\nClassification Report for {name} (Test Set):")
        print(report)
        print("-" * 60)
        
    # Final Reporting
    results_df = pd.DataFrame(results_list).sort_values(by='Accuracy', ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'baseline_results_pro.csv'), index=False)
    
    plot_matrices(confusion_matrices, class_names)
    
    logger.info("=== FINAL TEST SET PERFORMANCE ===")
    print("\n" + results_df.to_string(index=False))

if __name__ == "__main__":
    main()