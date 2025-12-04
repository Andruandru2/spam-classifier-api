'''from flask import Flask, request, jsonify
import joblib
import traceback

app = Flask(__name__)

# -------------------------------
# Load Model + Vectorizer
# -------------------------------
try:
    model = joblib.load("best_spam_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model or vectorizer: {e}")

# -------------------------------
# Root Route
# -------------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "running",
        "message": "Spam Classifier API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }), 200

# -------------------------------
# Health Check
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# -------------------------------
# Predict Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate JSON
        if not data:
            return jsonify({"error": "JSON body missing"}), 400

        if "message" not in data:
            return jsonify({"error": "'message' field missing"}), 400

        message = str(data["message"]).strip()

        # Validate message content
        if len(message) < 3:
            return jsonify({"error": "Message too short"}), 400

        # Transform
        vect = vectorizer.transform([message]).toarray()

        # Predict
        prediction = model.predict(vect)[0]

        # Confidence score
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(vect)[0][prediction])
        else:
            confidence = None

        return jsonify({
            "message": message,
            "prediction": "spam" if prediction == 1 else "not_spam",
            "flag": int(prediction),
            "confidence": confidence
        }), 200

    except Exception as e:
        # Return clean error instead of Flask HTML error
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "trace": traceback.format_exc()
        }), 500


# -------------------------------
# Start Server
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)'''


# app.py
'''import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Config - if app.py is in same dir as the model files, this works:
MODEL_DIR = os.path.dirname(__file__)  # folder containing app.py
MODEL_PATH = os.path.join(MODEL_DIR, "best_spam_model.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# Helper: return confidence when available
def get_confidence(m, X_vec):
    # logistic and naive_bayes have predict_proba; LinearSVC does not
    try:
        probs = m.predict_proba(X_vec)
        # return probability of class 1 (spam)
        return float(probs[0][1])
    except Exception:
        # fallback: use decision_function if available, then map to [0,1] with sigmoid
        try:
            score = m.decision_function(X_vec)[0]
            # sigmoid
            conf = 1 / (1 + np.exp(-score))
            return float(conf)
        except Exception:
            return None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message":"Spam classifier API is up"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = None
    if request.is_json:
        data = request.get_json()
    else:
        # also accept form-data fallback
        data = request.form.to_dict()

    if not data or "message" not in data:
        return jsonify({"error":"Missing 'message' field"}), 400

    message = data["message"]
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    conf = get_confidence(model, vec)

    return jsonify({
        "message": message,
        "prediction": "spam" if int(pred)==1 else "not_spam",
        "flag": int(pred),
        "confidence": conf
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    #without fast api'''

# With fast api
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

# ---------- Load model ----------
MODEL_PATH = "./model"

tokenizer = DistilBertTokenizerFast.from_pretrained("Andrues/my-model/model")
model = DistilBertForSequenceClassification.from_pretrained("Andrues/my-model/model")


#tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
#model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# ---------- FastAPI App ----------
app = FastAPI(title="Spam Classifier API", version="1.0")

class Message(BaseModel):
    text: str

def predict_spam(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

    return pred_class, confidence

# ---------- Routes ----------
@app.get("/")
def home():
    return {"status": "Transformer Spam API running"}

@app.post("/predict")
def predict(msg: Message):
    label, conf = predict_spam(msg.text)
    
    return {
        "input": msg.text,
        "prediction": "spam" if label == 1 else "ham",
        "flag": label,
        "confidence": round(conf, 4)
    }







