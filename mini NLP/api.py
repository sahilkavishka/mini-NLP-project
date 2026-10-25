from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from collections import Counter
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./bert_outputs/bert_best_model_production"

print("Loading AI Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
print("Server is Ready!")

class ReviewData(BaseModel):
    reviews: list[str]

@app.post("/analyze")
def analyze_reviews(data: ReviewData):
    if not data.reviews:
        return {"error": "No reviews provided"}
        
    
    inputs = tokenizer(data.reviews, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        
    predictions = torch.argmax(probs, dim=-1).tolist()
    
    positive_count = predictions.count(2)
    neutral_count = predictions.count(1)
    negative_count = predictions.count(0)
    

    stop_words = {
        "the", "and", "is", "it", "to", "a", "this", "for", "of", "in", "i", "my", "phone", 
        "with", "on", "was", "very", "so", "but", "eka", "me", "nam", "ek", "godak", 
        "thiyenawa", "seller", "received", "delivery", "product", "order", "came", "got", 
        "daraz", "time", "parcel", "from", "good", "nice", "not", "bad", "are", "have", 
        "has", "they", "you", "that", "can", "will", "always", "really", "just", "like", 
        "too", "much", "well", "also", "only", "even", "out", "about", "get", "bought", 
        "buy", "give", "item", "one", "all", "after", "use", "using", "when", "then"
    }
    
    
    tech_aspects = {
        "battery", "camera", "display", "screen", "performance", "price", "quality", 
        "design", "speed", "charge", "charging", "sound", "speaker", "fingerprint", 
        "storage", "ram", "processor", "build", "color", "service", "warranty", 
        "heating", "lag", "software", "update", "value", "money", "packing", "packaging", "glass"
    }
    
    positive_words = []
    negative_words = []

    
    for review, pred in zip(data.reviews, predictions):
        
        words = set(re.findall(r'\b[a-z]{4,}\b', review.lower()))
        filtered_words = [w for w in words if w not in stop_words]
        
        
        meaningful_words = [w for w in filtered_words if w in tech_aspects]
        
        
        if not meaningful_words:
            meaningful_words = filtered_words
            
        if pred == 2: # Positive review
            positive_words.extend(meaningful_words)
        elif pred == 0: # Negative review
            negative_words.extend(meaningful_words)

    
    total_analyzed = positive_count + negative_count + neutral_count
    if total_analyzed == 0:
        positive_percent = 50
    else:
        positive_percent = int((positive_count / total_analyzed) * 100)
        
    if (positive_count + negative_count) > 0:
        negative_percent = int((negative_count / total_analyzed) * 100)
    else:
        negative_percent = 0

    
    top_pros = [word for word, count in Counter(positive_words).most_common(4)]
    top_cons = [word for word, count in Counter(negative_words).most_common(4)]

    return {
        "positive_percentage": positive_percent,
        "negative_percentage": negative_percent, 
        "total_analyzed": len(data.reviews),
        "top_pros": top_pros,
        "top_cons": top_cons
    }
