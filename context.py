from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients
import torch
import torch.nn.functional as F
import os
import re

app = Flask(__name__)

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define models and label mappings
models_info = {
    "flag-bert": {
        "path": r"hub/flag-bert",
        "labels": {0: "unflag", 1: "flag"}
    },
    "profanity-bertweet": {
        "path": r"hub/profanity-bertweet",
        "labels": {0: "not_profanity", 1: "profanity"}
    },
    "roberta-offensive": {
        "path": r"hub/roberta-offensive",
        "labels": {0: "hate speech", 1: "non-offensive", 2: "offensive"}
    },
    "bert-hatespeech": {
        "path": r"hub/hatespeech-bert",
        "labels": {0: "age", 1: "ethnicity", 2: "gender", 3: "not_cyberbullying", 4: "other_cyberbullying", 5: "religion"}
    },
    "roberta-sentiment": {
        "path": r"hub/roberta-sentiment",
        "labels": {0: "negative", 1: "neutral", 2: "positive"}
    }
}

# ✅ Load models and tokenizers
models, tokenizers = {}, {}

for model_name, info in models_info.items():
    model_path = info["path"]
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        continue
    tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
    models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

# ✅ Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

# ✅ Clean tokens properly before displaying
def clean_token(token):
    return token.replace("▁", "").replace("Ġ", "").replace("</s>", "").replace("_", "").strip()

# ✅ Custom forward function for IG
def custom_forward(inputs_embeds, attention_mask, model):
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

# ✅ Analyze text and get predictions, confidence, and attributions
def analyze_text(text, model_name):
    model, tokenizer = models[model_name], tokenizers[model_name]
    labels = models_info[model_name]["labels"]
    
    # Clean and tokenize text
    text = clean_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

    # ✅ Get logits and probabilities
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = F.softmax(logits, dim=1)
    predicted_index = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_index].item() * 100
    predicted_label = labels[predicted_index]

    # ✅ Compute Integrated Gradients for attributions
    inputs_embeds = model.roberta.embeddings.word_embeddings(input_ids)
    ig = IntegratedGradients(lambda inputs_embeds, attention_mask: custom_forward(inputs_embeds, attention_mask, model))
    attributions = ig.attribute(inputs_embeds, baselines=torch.zeros_like(inputs_embeds).to(device),
                                 target=predicted_index, additional_forward_args=(attention_mask,))
    attributions = attributions.sum(dim=-1).squeeze(0)

    # ✅ Map tokens to attribution scores
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    word_importance = [(clean_token(token), score)
                       for token, score in zip(tokens, attributions.tolist())
                       if token not in ["[CLS]", "[SEP]", "[PAD]"]]
    word_importance = sorted(word_importance, key=lambda x: abs(x[1]), reverse=True)

    return predicted_label, confidence, word_importance

# ✅ Combine consecutive influential words into phrases
def combine_phrases(word_importance, threshold=0.1):
    phrases, current_phrase, current_score = [], [], 0
    for token, score in word_importance:
        clean_word = clean_token(token)
        if clean_word and abs(score) > threshold:
            current_phrase.append(clean_word)
            current_score += score
        elif current_phrase:
            phrases.append((" ".join(current_phrase), current_score))
            current_phrase, current_score = [], 0
    if current_phrase:
        phrases.append((" ".join(current_phrase), current_score))
    
    # ✅ Format phrases properly
    phrase_list = [
        {"phrase": phrase, "score": round(score, 2), "influence": "Flagged" if score > 0 else "Not Flagged"}
        for phrase, score in phrases
    ]
    return phrase_list

# ✅ Display cleaned word scores with emojis
def display_word_scores_emoji(word_importance, threshold=0.0):
    """Display cleaned word scores with emojis and influence type."""
    ignore_tokens = {"", "<s>", "</s>", "[SEP]", "[CLS]", "[PAD]"}
    summary = []
    for token, score in word_importance:
        clean_word = clean_token(token)
        if clean_word and clean_word not in ignore_tokens and abs(score) > threshold:
            emoji = "📈" if score > 0 else "📉"
            influence_type = "Flagged" if score > 0 else "Not Flagged"
            summary.append({
                "word": clean_word,
                "score": round(score, 4),
                "influence": influence_type,
                "emoji": emoji
            })
    return summary

# ✅ Main predict function for all models with attribution
def predict_with_explanation(text, model_name):
    predicted_label, confidence, word_importance = analyze_text(text, model_name)
    word_summary = display_word_scores_emoji(word_importance, threshold=0.1)
    phrase_summary = combine_phrases(word_importance, threshold=0.1)
    
    return {
        "label": predicted_label,
        "probability": f"{confidence:.2f}%",
        "important_words": word_summary,
        "influential_phrases": phrase_summary
    }

### **🔹 Updated API Endpoints with Attribution**

@app.route('/flag', methods=['POST'])
def predict_flag_bert():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(predict_with_explanation(text, "flag-bert"))

@app.route('/profanity', methods=['POST'])
def predict_profanity_bertweet():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(predict_with_explanation(text, "profanity-bertweet"))

@app.route('/offensive', methods=['POST'])
def predict_roberta_offensive():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(predict_with_explanation(text, "roberta-offensive"))

@app.route('/hatespeech', methods=['POST'])
def predict_hatespeech_bert():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(predict_with_explanation(text, "bert-hatespeech"))

@app.route('/sentiment', methods=['POST'])
def predict_roberta_sentiment():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(predict_with_explanation(text, "roberta-sentiment"))

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(host="172.20.10.8", port=5000, debug=True)
