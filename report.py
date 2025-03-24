from flask import Flask, request, jsonify, send_file
import requests
from fpdf import FPDF
from datetime import datetime

app = Flask(__name__)

# ✅ Define the endpoints for each AI model
FLAG_BERT_URL = "http://10.226.54.4:5000/flag"
PROFANITY_URL = "http://10.226.54.4:5000/profanity"
OFFENSIVE_URL = "http://10.226.54.4:5000/offensive"
HATE_SPEECH_URL = "http://10.226.54.4:5000/hatespeech"
SENTIMENT_URL = "http://10.226.54.4:5000/sentiment"


# ✅ Generate influence summary from phrases
def format_influence_summary(influence_phrases):
    if not influence_phrases or len(influence_phrases) == 0:
        return "No influential phrases detected."
    phrase_summary = []
    for phrase in influence_phrases:
        score = round(float(phrase["score"]) * 100, 2)
        influence = phrase["influence"]
        phrase_summary.append(
            f"- The phrase '{phrase['phrase']}' was {influence.lower()} with a score impact of {score}%."
        )
    return "\n".join(phrase_summary)


# ✅ Generate detailed sentiment and influence summary
def generate_detailed_summary(response, label, probability, influential_phrases):
    phrase_summary = format_influence_summary(influential_phrases)
    summary = (
        f"The post is inclined towards {'a Positive sentiment' if label == 'positive' else 'a Negative sentiment' if label == 'negative' else 'a Neutral sentiment'}, "
        f"with a confidence of {probability}.\n"
        f"{'The phrase \"' + influential_phrases[0]['phrase'] + '\" had the most significant impact on the classification, increasing the likelihood by ' + str(round(influential_phrases[0]['score'] * 100, 2)) + '%.' if influential_phrases else 'No specific phrases significantly influenced the classification.'}\n\n"
        f"Detailed Insights:\n"
        f"- Sentiment Label: {label.capitalize()} ({probability})\n"
        f"- Influence Summary: {'A flagged phrase was detected, indicating potential concern.' if any(p['influence'] == 'Flagged' for p in influential_phrases) else 'No flagged phrases detected.'}\n"
        f"- Recommendation: {'Consider further review due to potential flagged content.' if label == 'negative' else 'No immediate action required.'}"
    )
    return summary


# ✅ Generate PDF report
def generate_pdf(report_sections, pdf_file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    # Title
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, "Social Media Post Analysis Report", ln=True, align="C")
    pdf.ln(5)

    # Add sections to PDF
    pdf.set_font("Arial", size=12)
    for heading, content in report_sections:
        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, heading)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, content)
        pdf.ln(2)

    # Save the PDF
    pdf.output(pdf_file_path)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")
    post_id = data.get("post_id", "123456")
    post_link = data.get("post_link", "")
    user_source = data.get("user_id", "@user123")
    platform = data.get("platform", "Twitter")
    timestamp = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %I:%M %p"))

    # ✅ Step 1: Flag the text
    flag_response = requests.post(FLAG_BERT_URL, json={"text": text}).json()
    if flag_response["label"] == "not_flagged":
        return jsonify({"status": "Safe", "analysis": flag_response})

    # ✅ Step 2: Perform further analysis
    sentiment_response = requests.post(SENTIMENT_URL, json={"text": text}).json()
    profanity_response = requests.post(PROFANITY_URL, json={"text": text}).json()
    offensive_response = requests.post(OFFENSIVE_URL, json={"text": text}).json()
    hate_speech_response = requests.post(HATE_SPEECH_URL, json={"text": text}).json()

    # ✅ Extract probabilities
    offensive_prob = float(offensive_response["probability"].strip('%')) / 100
    profanity_prob = float(profanity_response["probability"].strip('%')) / 100
    hate_speech_prob = float(hate_speech_response["probability"].strip('%')) / 100
    sentiment_prob = float(sentiment_response["probability"].strip('%')) / 100

    # ✅ Get influential phrases for each model
    sentiment_summary = generate_detailed_summary(
        sentiment_response,
        sentiment_response["label"],
        sentiment_response["probability"],
        sentiment_response.get("influential_phrases", []),
    )
    profanity_summary = format_influence_summary(profanity_response.get("influential_phrases", []))
    offensive_summary = format_influence_summary(offensive_response.get("influential_phrases", []))
    hate_speech_summary = format_influence_summary(hate_speech_response.get("influential_phrases", []))

    # ✅ Define weights for severity calculation
    HATE_WEIGHT = 0.5
    OFFENSIVE_WEIGHT = 0.3
    PROFANITY_WEIGHT = 0.2

    # ✅ Calculate severity score
    severity_score = (
        (hate_speech_prob * HATE_WEIGHT) +
        (offensive_prob * OFFENSIVE_WEIGHT) +
        (profanity_prob * PROFANITY_WEIGHT)
    )

    # ✅ Define severity levels
    if severity_score > 0.8:
        severity = "Immediate Takedown"
    elif severity_score > 0.6:
        severity = "High"
    elif severity_score > 0.4:
        severity = "Medium"
    elif severity_score > 0.2:
        severity = "Low"
    else:
        severity = "Moderate"

    # ✅ Prepare the enhanced report
    report_sections = [
        (
            "Post Details:",
            f"Post ID: {post_id}\nUser/Source: {user_source} (Platform: {platform})\nDate & Time: {timestamp}\nPost Content:\n\"{text}\"",
        ),
        ("Sentiment Analysis:", sentiment_summary),
        (
            "Profanity Detection:",
            f"Profanity Classification: {profanity_response['label']}\n"
            f"Profanity Probability: {profanity_prob * 100}%\n\n"
            f"Influence Analysis:\n{profanity_summary}",
        ),
        (
            "Offensive Language Detection:",
            f"Offensive Language Classification: {offensive_response['label']}\n"
            f"Offensive Language Probability: {offensive_prob * 100}%\n\n"
            f"Influence Analysis:\n{offensive_summary}",
        ),
        (
            "Hate Speech Detection:",
            f"Hate Speech Classification: {hate_speech_response['label']}\n"
            f"Hate Speech Probability: {hate_speech_prob * 100}%\n\n"
            f"Influence Analysis:\n{hate_speech_summary}",
        ),
        (
            "Combined Analysis:",
            f"Overall Post Category: {'🚨Alert' if severity == 'Immediate Takedown' else 'Hate Speech' if severity == 'High' else 'Offensive Language' if severity == 'Medium' else 'Profanity' if severity == 'Low' else 'For review'}\n"
            f"Action Recommendation: {severity}\n"
            f"The post is marked as {'Flagged' if flag_response['label'] == 'flag' else 'Not Flagged'}.",
        ),
        (
            "Weight Representation:",
            f"Offensive Language Probability: {offensive_prob * 100}%\n"
            f"Profanity Probability: {profanity_prob * 100}%\n"
            f"Hate Speech Probability: {hate_speech_prob * 100}%",
        ),
        (
            "Action Recommendation:",
            f"Suggestion: Given these factors, this post is marked as {severity if severity == 'Immediate Takedown' else 'flagged for review'}.",
        ),
    ]

    # ✅ Generate and send PDF report
    pdf_file_path = "analysis_report.pdf"
    generate_pdf(report_sections, pdf_file_path)

    return send_file(pdf_file_path, as_attachment=True)


# ✅ Run Flask App
if __name__ == "__main__":
    app.run(host="10.226.51.33", port=5000, debug=False)