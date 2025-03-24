from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import os

app = Flask(__name__)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define models and their corresponding label mappings
models_info = {
    "flag-bert": {
        "path": r"hub\flag-bert",
        "labels": {0: "unflag", 1: "flag"}
    },

    "profanity-bertweet": {
        "path": r"hub\profanity-bertweet",
        "labels": {0: "not_profanity", 1: "profanity"}
    },

    "roberta-offensive": {
        "path": r"hub\roberta-offensive",
        "labels": {0: "hate speech", 1: "non-offensive", 2: "offensive"}
    },

    "bert-hatespeech": {
        "path": r"hub/hatespeech-bert",
        "labels": {0: "age", 1: "ethnicity", 2: "gender", 3: "not_cyberbullying", 4: "other_cyberbullying", 5: "religion"}
    },

    "roberta-sentiment": {
        "path": r"hub\roberta-sentiment",
        "labels": {0: "negative", 1: "neutral", 2: "positive"}
    }
}

# Load models and tokenizers
models = {}
tokenizers = {}

for model_name, info in models_info.items():
    model_path = info["path"]
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        continue

    print(f"Loading {model_name} from {model_path}...")
    tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
    models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

# Function to run inference
def predict(text, model_name):
    tokenizer = tokenizers[model_name]
    model = models[model_name]
    labels = models_info[model_name]["labels"]

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Compute probabilities
    probabilities = F.softmax(logits, dim=1)
    predicted_index = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
    predicted_prob = probabilities[0, predicted_index].cpu().item() * 100

    # Get label text
    predicted_label = labels[predicted_index]

    return {"label": predicted_label, "probability": f"{predicted_prob:.2f}%"}


### **🔹 Separate API Endpoints for Each Model**

@app.route('/flag', methods=['POST'])
def predict_flag_bert():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(predict(text, "flag-bert"))

@app.route('/profanity', methods=['POST'])
def predict_profanity_bertweet():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(predict(text, "profanity-bertweet"))

@app.route('/offensive', methods=['POST'])
def predict_roberta_offensive():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(predict(text, "roberta-offensive"))

@app.route('/hatespeech', methods=['POST'])
def predict_hatespeech_bert():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(predict(text, "bert-hatespeech"))

@app.route('/sentiment', methods=['POST'])
def predict_roberta_sentiment():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(predict(text, "roberta-sentiment"))


# Run Flask App
if __name__ == '__main__':
    app.run(host="172.20.10.8", port=5000, debug=True)
