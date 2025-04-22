import os
import pandas as pd
from flask import Flask, request, render_template, send_file, flash
from transformers import pipeline

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

UPLOAD_FOLDER = 'uploads'
HASIL_FOLDER = 'hasil'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HASIL_FOLDER, exist_ok=True)

sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def convert_label(label):
    if label in ['1 star', '2 stars']:
        return 'Negatif'
    elif label == '3 stars':
        return 'Netral'
    else:
        return 'Positif'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/proses', methods=['POST'])
def proses():
    if 'csvfile' not in request.files:
        flash("No file part", "error")
        return render_template('index.html')

    file = request.files['csvfile']

    if file.filename == '':
        flash("No selected file", "error")
        return render_template('index.html')

    if not file.filename.endswith('.csv'):
        flash("Invalid file type. Please upload a CSV file.", "error")
        return render_template('index.html')

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        flash(f"Error reading the CSV file: {e}", "error")
        return render_template('index.html')

    if 'full_text' not in df.columns:
        flash("CSV file must contain a 'full_text' column", "error")
        return render_template('index.html')

    df['full_text'] = df['full_text'].astype(str)
    sentiments = sentiment_pipeline(df['full_text'].tolist())

    df["sentiment_label_raw"] = [s["label"] for s in sentiments]
    df["sentiment_score"] = [s["score"] for s in sentiments]
    df["sentiment"] = df["sentiment_label_raw"].apply(convert_label)

    hasil_path = os.path.join(HASIL_FOLDER, "hasil_" + file.filename)
    df.to_csv(hasil_path, index=False)

    return send_file(hasil_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
