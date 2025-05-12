import os
import pandas as pd
import joblib
import json
import re
from google_play_scraper import reviews, Sort
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from math import ceil
from youtube_comment_downloader import YoutubeCommentDownloader
from urllib.parse import urlparse, parse_qs


app = Flask(__name__)  # Inisialisasi aplikasi Flask
app.secret_key = 'rahasia-super-aman-123'  # Kunci rahasia untuk session (misal: flash message)

UPLOAD_FOLDER = 'uploads'  # Folder untuk menyimpan file yang diupload user
HASIL_FOLDER = 'hasil'     # Folder untuk menyimpan file hasil setelah diproses (misal hasil analisis sentimen)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Bikin folder uploads jika belum ada
os.makedirs(HASIL_FOLDER, exist_ok=True)   # Bikin folder hasil jika belum ada

# Load Model IndoBERT RoBERTa Sentiment Bahasa Indonesia
sentiment_pipeline = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")

def bersihkan_teks(teks):
    # Emoji ke kata
    emoji_dict = {
        r'😂|🤣|😹': ' lucu ',
        r'😊|😁|😄|🙂|😃|😆|😸|☺️': ' senang ',
        r'😍|🥰|😘|😻': ' cinta ',
        r'😢|😭|😞|😔|☹️|😿': ' sedih ',
        r'😡|😠|🤬': ' marah ',
        r'🤔|😕|😐|😑': ' bingung ',
        r'😴|🥱|😪': ' ngantuk ',
        r'👍|👌|✌️|👏': ' bagus ',
        r'👎|🙄|😒': ' buruk ',
    }
    for pattern, replacement in emoji_dict.items():
        teks = re.sub(pattern, replacement, teks)
    
    # Normalisasi kata gaul & slang
    slang_dict = {
        r'\bmantap\b|\bmantul\b|\bmantep\b|\bgokil\b': 'bagus',
        r'\bzonk\b|\bampas\b|\bburikk\b': 'buruk',
        r'\bbt\b|\bbete\b': 'sedih',
        r'\bluar biasa\b': 'bagus',
        r'\banjg\b|\bajg\b|\bkontol\b': 'kasar',
        r'\bgpp\b': 'gak apa-apa',
        r'\bmager\b': 'malas gerak',
        r'\bngab\b': 'teman',
        r'\bciyus\b': 'serius',
        r'\bseblak\b': 'makanan pedas',
        r'\bngopi\b': 'minum kopi',
    }
    for slang, normal in slang_dict.items():
        teks = re.sub(slang, normal, teks)
    
    # Singkatan informal
    singkatan_dict = {
        r'\bgk\b': 'gak',
        r'\bga\b': 'gak',
        r'\btp\b': 'tapi',
        r'\bdgn\b': 'dengan',
        r'\bdr\b': 'dari',
        r'\bdg\b': 'dengan',
    }
    for sgk, normal in singkatan_dict.items():
        teks = re.sub(sgk, normal, teks)

    # Hilangkan URL, mention, hashtag, angka
    teks = re.sub(r'http\S+|www\S+', '', teks)  # url
    teks = re.sub(r'@\w+', '', teks)  # mention
    teks = re.sub(r'#\w+', '', teks)  # hashtag
    teks = re.sub(r'\d+', '', teks)  # angka

    # Hilangkan simbol spam
    teks = re.sub(r'!{2,}', '!', teks)
    teks = re.sub(r'\?{2,}', '?', teks)
    teks = re.sub(r'\s+', ' ', teks)

    # Lowercase & trim
    teks = teks.lower().strip()
    return teks


# Konversi label + threshold score biar lebih akurat
def convert_label_with_threshold(label, score):
    if score < 0.6:
        return 'Netral'
    elif label.lower() == 'positive':
        return 'Positif'
    elif label.lower() == 'negative':
        return 'Negatif'
    else:
        return 'Netral'

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

    # Bersihkan teks dari emoticon & kata gaul
    df['full_text'] = df['full_text'].astype(str).apply(bersihkan_teks)

    # Prediksi sentimen
    sentiments = sentiment_pipeline(df['full_text'].tolist())

    # Masukkan hasil ke dataframe
    df["sentiment_label_raw"] = [s["label"] for s in sentiments]
    df["sentiment_score"] = [s["score"] for s in sentiments]

    # Konversi label final dengan threshold
    df["sentiment"] = [
        convert_label_with_threshold(l, s) for l, s in zip(df["sentiment_label_raw"], df["sentiment_score"])
    ]

    # Simpan hasil CSV
    hasil_filename = "hasil_" + file.filename
    hasil_path = os.path.join(HASIL_FOLDER, hasil_filename)
    df.to_csv(hasil_path, index=False)

    return send_file(
        hasil_path,
        as_attachment=True,
        download_name=hasil_filename,
        mimetype='text/csv'
    )

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
    hasil_path = os.path.join(HASIL_FOLDER, "last_result.csv")
    report_path = os.path.join(HASIL_FOLDER, "report.json")

    if not os.path.exists(hasil_path) or not os.path.exists(report_path):
        flash("Belum ada hasil analisis. Silakan unggah dan analisis file terlebih dahulu.", "error")
        return redirect(url_for('index'))

    # Baca data hasil dan metadata
    hasil_df = pd.read_csv(hasil_path)
    with open(report_path, "r") as f:
        meta = json.load(f)

    # Ambil filter dan page dari URL
    filter_value = request.args.get('filter', '').strip().lower()
    page = int(request.args.get('page', 1))
    per_page = 10

    # Filter berdasarkan kolom 'predicted' jika filter aktif
    if filter_value:
        hasil_df = hasil_df[hasil_df['predicted'].str.lower().str.contains(filter_value)]

    # Hitung total data dan pagination
    total_data = len(hasil_df)
    total_pages = ceil(total_data / per_page) if total_data > 0 else 1
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
                           total_pages=total_pages,
                           filter_value=filter_value)


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

        # Kirim file dengan MIME type eksplisit agar tidak dikenali sebagai .xls
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )

    except Exception as e:
        flash(f"Gagal mengambil ulasan: {e}", "error")
        return redirect(url_for('index'))

def extract_video_id(yt_url):
    parsed_url = urlparse(yt_url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    elif parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query = parse_qs(parsed_url.query)
        return query.get("v", [None])[0]
    return None

@app.route('/ambil_komentar_yt', methods=['POST'])
def ambil_komentar_yt():
    url = request.form.get('yt_url')

    if not url:
        flash("URL YouTube tidak ditemukan", "error")
        return redirect(url_for('index'))

    # Ekstrak video ID dari URL
    query = urlparse(url)
    video_id = None
    if query.hostname in ['www.youtube.com', 'youtube.com']:
        video_id = parse_qs(query.query).get("v", [None])[0]
    elif query.hostname == 'youtu.be':
        video_id = query.path[1:]

    if not video_id:
        flash("Gagal mengambil Video ID dari URL", "error")
        return redirect(url_for('index'))

    try:
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}", sort_by="top")

        hasil_komentar = []
        for comment in comments:
            hasil_komentar.append({
                "user": comment["author"],
                "comment": comment["text"],
                "likes": comment["votes"]
            })

        # Simpan ke file CSV
        df = pd.DataFrame(hasil_komentar)
        filename = f"yt_komentar_{video_id}.csv"
        path = os.path.join(HASIL_FOLDER, filename)
        df.to_csv(path, index=False)

        return send_file(
            path,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )

    except Exception as e:
        flash(f"Gagal mengambil komentar: {str(e)}", "error")
        return redirect(url_for('index'))


@app.route('/analisis')
def analisis():
    return render_template('analisis.html')

@app.route('/panduan')
def panduan():
    return render_template('panduan.html')
    
@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_message = data.get('message', '').lower()

    responses = {
        "halo": "Halo juga! Ada yang bisa saya bantu terkait analisis sentimen? 😊",
        "apa itu analisis sentimen": "Analisis sentimen adalah proses mengidentifikasi emosi atau opini dalam teks, seperti positif, netral, atau negatif.",
        "model apa yang digunakan": "Kami menggunakan dua model: Naive Bayes untuk teks sederhana, dan BERT multilingual dari Hugging Face untuk analisis mendalam.",
        "bagaimana cara mengunggah file csv": "Silakan klik 'Pilih File' di halaman utama, lalu pilih file CSV yang berisi kolom 'full_text'. Setelah itu klik 'Proses'.",
        "apa itu fitur proses": "Fitur 'Proses' menganalisis teks ulasan menggunakan model BERT dan mengkategorikan menjadi Positif, Netral, atau Negatif.",
        "apa itu fitur analyze": "Fitur 'Analyze CSV' digunakan untuk melatih model Naive Bayes dari data yang Anda upload, setelah balancing otomatis.",
        "apa itu fitur hasil": "Fitur 'Hasil' menampilkan ringkasan performa model, seperti akurasi, precision, recall, dan confusion matrix.",
        "bagaimana cara download ulasan play store": "Gunakan form 'Download Ulasan', masukkan App ID dari Play Store, lalu klik 'Download'. File CSV akan otomatis dibuat.",
        "bagaimana cara klasifikasi satu ulasan": "Anda bisa mengetikkan satu ulasan di kolom 'Klasifikasi Manual' dan mendapatkan hasil prediksi sentimennya.",
        "apa itu fitur chatbot": "Fitur chatbot ini membantu Anda memahami fungsi-fungsi yang tersedia dalam aplikasi analisis sentimen ini.",
        "terima kasih": "Sama-sama! Jika ada lagi yang ingin ditanyakan, jangan ragu. 😊",
        "siapa kamu": "Saya adalah chatbot pendamping untuk aplikasi analisis sentimen berbasis Flask yang Anda gunakan."
    }

    # Menu bantuan
    if user_message in ["menu", "bantuan", "help"]:
        response = (
            "Berikut beberapa pertanyaan yang bisa Anda ajukan:\n"
            "1. Apa itu analisis sentimen\n"
            "2. Model apa yang digunakan\n"
            "3. Bagaimana cara mengunggah file CSV\n"
            "4. Apa itu fitur proses\n"
            "5. Apa itu fitur analyze\n"
            "6. Apa itu fitur hasil\n"
            "7. Bagaimana cara download ulasan Play Store\n"
            "8. Bagaimana cara klasifikasi satu ulasan\n"
            "9. Apa itu fitur chatbot\n"
            "10. Siapa kamu\n"
            "Silakan ketik salah satu pertanyaan di atas untuk penjelasan lebih lanjut. 😊"
        )
    else:
        # Ambil respons dari dict atau default
        response = responses.get(user_message, "Maaf, saya belum mengerti pertanyaan itu. Coba ketik 'menu' untuk melihat daftar bantuan.")

    return {"response": response}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
