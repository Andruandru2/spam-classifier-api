from flask import Flask, request, jsonify
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
    app.run(host="0.0.0.0", port=10000)
