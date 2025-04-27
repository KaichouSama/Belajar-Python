import os
import pandas as pd
import joblib
from flask import Response
from google_play_scraper import reviews, Sort
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from math import ceil

app = Flask(__name__)
app.secret_key = 'rahasia-super-aman-123'

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

@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    file = request.files['file']
    if file.filename == '':
        flash("No selected file", "error")
        return redirect(url_for('index'))

    df = pd.read_csv(file)
    if 'full_text' not in df.columns or 'sentiment' not in df.columns:
        flash("CSV must contain 'full_text' and 'sentiment'", "error")
        return redirect(url_for('index'))

    distribusi_awal = df['sentiment'].value_counts()
    balance_before_split = 'balance_before_split' in request.form

    if balance_before_split:
        # === Balancing sebelum split ===
        kelas = df['sentiment'].unique()
        max_count = df['sentiment'].value_counts().max()

        df_balanced = pd.concat([
            resample(df[df['sentiment'] == k], replace=True, n_samples=max_count, random_state=42)
            for k in kelas
        ])

        distribusi_balanced = df_balanced['sentiment'].value_counts()

        X_train, X_test, y_train, y_test = train_test_split(
            df_balanced["full_text"], df_balanced["sentiment"], test_size=0.2, random_state=42, stratify=df_balanced["sentiment"]
        )

    else:
        # === Balancing setelah split ===
        X_train, X_test, y_train, y_test = train_test_split(
            df["full_text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
        )

        train_df = pd.DataFrame({'full_text': X_train, 'sentiment': y_train})
        kelas = train_df['sentiment'].unique()
        max_count = train_df['sentiment'].value_counts().max()

        train_df_balanced = pd.concat([
            resample(train_df[train_df['sentiment'] == k], replace=True, n_samples=max_count, random_state=42)
            for k in kelas
        ])

        distribusi_balanced = train_df_balanced['sentiment'].value_counts()

        X_train = train_df_balanced['full_text']
        y_train = train_df_balanced['sentiment']

    # === Mulai training ===
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    akurasi = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    hasil_df = pd.DataFrame({
        "full_text": X_test,
        "actual": y_test,
        "predicted": y_pred
    })
    hasil_df.to_csv(os.path.join(HASIL_FOLDER, "last_result.csv"), index=False)

    report_path = os.path.join(HASIL_FOLDER, "report.json")
    with open(report_path, "w") as f:
        import json
        json.dump({
            "distribusi_awal": distribusi_awal.to_dict(),
            "distribusi_balanced": distribusi_balanced.to_dict(),
            "akurasi": akurasi,
            "report": report,
            "matrix": matrix.tolist()
        }, f)

    joblib.dump(vectorizer, os.path.join(HASIL_FOLDER, "vectorizer.pkl"))
    joblib.dump(model, os.path.join(HASIL_FOLDER, "model.pkl"))

    return redirect(url_for('hasil', page=1))

@app.route('/hasil')
def hasil():
    import json

    hasil_path = os.path.join(HASIL_FOLDER, "last_result.csv")
    report_path = os.path.join(HASIL_FOLDER, "report.json")

    if not os.path.exists(hasil_path) or not os.path.exists(report_path):
        flash("Belum ada hasil analisis. Silakan unggah dan analisis file terlebih dahulu.", "error")
        return redirect(url_for('index'))

    hasil_df = pd.read_csv(hasil_path)
    with open(report_path, "r") as f:
        meta = json.load(f)

    page = int(request.args.get('page', 1))
    per_page = 10
    total_data = len(hasil_df)
    total_pages = ceil(total_data / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    hasil_page = hasil_df.iloc[start:end].to_dict(orient='records')

    return render_template('result.html',
                           distribusi_awal=meta["distribusi_awal"],
                           distribusi_balanced=meta["distribusi_balanced"],
                           akurasi=meta["akurasi"],
                           report=meta["report"],
                           matrix=meta["matrix"],
                           hasil=hasil_page,
                           page=page,
                           total_pages=total_pages)

@app.route('/klasifikasi', methods=['POST'])
def klasifikasi():
    ulasan = request.form.get('ulasan')
    if not ulasan:
        return render_template('result.html', hasil_klasifikasi="(ulasan kosong)")

    vectorizer_path = os.path.join(HASIL_FOLDER, "vectorizer.pkl")
    model_path = os.path.join(HASIL_FOLDER, "model.pkl")
    hasil_path = os.path.join(HASIL_FOLDER, "last_result.csv")
    report_path = os.path.join(HASIL_FOLDER, "report.json")

    if not (os.path.exists(vectorizer_path) and os.path.exists(model_path) and os.path.exists(hasil_path) and os.path.exists(report_path)):
        return render_template('result.html', hasil_klasifikasi="Model belum dilatih")

    import joblib, json
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    hasil_df = pd.read_csv(hasil_path)

    with open(report_path, 'r') as f:
        meta = json.load(f)

    hasil = model.predict(vectorizer.transform([ulasan]))[0]

    page = 1
    per_page = 10
    hasil_page = hasil_df.iloc[0:per_page].to_dict(orient='records')

    return render_template('result.html',
        hasil_klasifikasi=hasil,
        distribusi_awal=meta["distribusi_awal"],
        distribusi_balanced=meta["distribusi_balanced"],
        akurasi=meta["akurasi"],
        report=meta["report"],
        matrix=meta["matrix"],
        hasil=hasil_page,
        page=page,
        total_pages=ceil(len(hasil_df)/per_page))

@app.route('/download_ulasan', methods=['POST'])
def download_ulasan():
    app_id = request.form.get('appid')
    if not app_id:
        flash("App ID kosong.", "error")
        return redirect(url_for('index'))

    try:
        hasil_reviews, _ = reviews(
            app_id,
            lang='id',       # bahasa Indonesia
            country='id',    # negara Indonesia
            sort=Sort.NEWEST,
            count=1000       # ambil 1000 ulasan terbaru
        )

        # Convert ke DataFrame
        df = pd.DataFrame(hasil_reviews)

        # Membuat DataFrame baru
        df_simple = pd.DataFrame({
            'userName': df['userName'],
            'score': df['score'],
            'at': df['at'],
            'full_text': df['content']
        })

        # Menambahkan kolom kategori berdasarkan score
        def categorize(score):
            if score >= 4:
                return 'Positif'
            elif score == 3:
                return 'Netral'
            else:
                return 'Negatif'

        df_simple['kategori'] = df_simple['score'].apply(categorize)

        # Simpan ke CSV
        filename = f"ulasan_{app_id.replace('.', '_')}.csv"
        file_path = os.path.join(HASIL_FOLDER, filename)
        df_simple.to_csv(file_path, index=False)

        return send_file(file_path, as_attachment=True)

    except Exception as e:
        flash(f"Gagal mengambil ulasan: {e}", "error")
        return redirect(url_for('index'))

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_message = data.get('message', '').lower()

    # Contoh data dummy dengan respon sederhana
    responses = {
        "halo": "Halo juga! Ada yang bisa saya bantu terkait analisis sentimen?",
        "apa itu analisis sentimen": "Analisis sentimen adalah proses mengidentifikasi emosi atau opini dalam teks, seperti positif, netral, atau negatif.",
        "model apa yang digunakan": "Kami menggunakan model Naive Bayes dan juga BERT multilingual dari Hugging Face.",
        "terima kasih": "Sama-sama! 😊",
        "siapa kamu": "Saya adalah chatbot untuk membantu analisis sentimen pada data teks."
    }

    response = responses.get(user_message, "Maaf, saya belum mengerti pertanyaan itu. Coba yang lain ya!")

    return {"response": response}

if __name__ == '__main__':
    app.run(debug=True)