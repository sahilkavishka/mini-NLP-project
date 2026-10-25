import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

# 1. Professional Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Constants based on NLP Project Blueprint
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
OUTPUT_DIR = "./bert_outputs"

set_seed(42)

# --- 2. Custom Weighted Trainer (Handles Class Imbalance) ---
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(model.device)
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
        return (loss, outputs) if return_outputs else loss

def load_and_prepare_data():
    logger.info("Loading datasets for BERT...")
    train_df = pd.read_csv('train.csv')[['text_for_bert', 'label']]
    val_df = pd.read_csv('val.csv')[['text_for_bert', 'label']]
    test_df = pd.read_csv('test.csv')[['text_for_bert', 'label']]
    
    for df in [train_df, val_df, test_df]:
        df.rename(columns={'text_for_bert': 'text'}, inplace=True)
        df['text'] = df['text'].fillna("")
        
    ds = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test': Dataset.from_pandas(test_df)
    })
    
    labels = train_df['label'].values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float)
    logger.info(f"Computed Class Weights: {weights}")
    
    return ds, class_weights_tensor

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def plot_bert_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    cm_normalized = cm.astype('float') / row_sums
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, annot=True, fmt='.3f', cmap='viridis',
        xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1
    )
    plt.title('BERT Sequence Classification Confusion Matrix', fontsize=15, pad=15)
    plt.xlabel('Predicted Sentiment', fontsize=12)
    plt.ylabel('True Sentiment', fontsize=12)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, 'bert_confusion_matrix_pro.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"High-Res BERT Confusion matrix saved to: {plot_path}")

def main():
    dataset, class_weights_tensor = load_and_prepare_data()
    class_names = ['Negative', 'Neutral', 'Positive']
    
    logger.info(f"Initializing AutoTokenizer for {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LENGTH
        )
        
    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    logger.info("Initializing Pre-trained BERT Architecture...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(class_names)
    )
    
    # Absolute minimum, bulletproof TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        save_total_limit=1
    )
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor
    )
    
    logger.info("=== Commencing BERT Fine-Tuning Execution ===")
    trainer.train()
    
    logger.info("=== Running Inference on Test Dataset ===")
    predictions = trainer.predict(tokenized_datasets["test"])
    
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    
    logger.info("\n" + classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    plot_bert_confusion_matrix(y_true, y_pred, class_names)
    
    final_model_path = os.path.join(OUTPUT_DIR, "bert_best_model_production")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Production-ready model serialized to: {final_model_path}")

if __name__ == "__main__":
    main()