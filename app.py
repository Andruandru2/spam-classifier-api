from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("best_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/")
def home():
    return jsonify({"message": "API is running"}), 200

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    content = request.json
    if not content or "message" not in content:
        return jsonify({"error": "Message field missing"}), 400

    message = content["message"]
    vect = vectorizer.transform([message]).toarray()
    prediction = model.predict(vect)[0]

    return jsonify({
        "message": message,
        "prediction": "spam" if prediction == 1 else "not_spam",
        "flag": int(prediction)
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
