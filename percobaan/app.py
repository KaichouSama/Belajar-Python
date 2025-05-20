import os
import pandas as pd
import joblib
import json
import re
import csv
import time
from google_play_scraper import reviews, Sort
from flask import (
    Flask,
    Response,
    request,
    render_template,
    send_file,
    flash,
    redirect,
    url_for,
)
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from math import ceil
from googleapiclient.discovery import build
from dotenv import load_dotenv




app = Flask(__name__)
app.secret_key = (
    "rahasia-super-aman-123"  # Kunci rahasia untuk session (misal: flash message)
)

load_dotenv()
# API_KEY = "AIzaSyAsduxTF9_87lB0qarJwznkpNSqcbbIrL8"
# youtube = build("youtube", "v3", developerKey=API_KEY)

API_KEY = os.getenv("API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

MAX_COMMENTS = 1000
MAX_COMMENT_LENGTH = 512  # Batas panjang komentar yang akan diunduh


def get_youtube_service():
    return build("youtube", "v3", developerKey=API_KEY)


def extract_video_id(url):
    # Ambil ID video dari berbagai bentuk URL YouTube
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None


def get_comments(video_id, max_comments=1000):
    comments = []
    next_page_token = None

    while True:
        results = (
            youtube.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100,
                pageToken=next_page_token,
            )
            .execute()
        )

        for item in results.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comment_text = snippet["textDisplay"]

            if len(comment_text) > MAX_COMMENT_LENGTH:
                continue

            comment_data = {
                "text": comment_text,
                "author": snippet.get("authorDisplayName", ""),
                "published_at": snippet.get("publishedAt", ""),
                "like_count": snippet.get("likeCount", 0),
                "reply_count": item.get("snippet", {}).get("totalReplyCount", 0),
            }
            comments.append(comment_data)

            if len(comments) >= max_comments:
                return comments

        next_page_token = results.get("nextPageToken")
        if not next_page_token:
            break

    return comments


def get_video_info(video_id):
    result = youtube.videos().list(part="snippet", id=video_id).execute()

    if result["items"]:
        snippet = result["items"][0]["snippet"]
        return {
            "video_title": snippet["title"],
            "channel_title": snippet["channelTitle"],
        }
    else:
        return {"video_title": "Unknown Title", "channel_title": "Unknown Channel"}


UPLOAD_FOLDER = "uploads"  # Folder untuk menyimpan file yang diupload user
HASIL_FOLDER = "hasil"  # Folder untuk menyimpan file hasil setelah diproses (misal hasil analisis sentimen)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Bikin folder uploads jika belum ada
os.makedirs(HASIL_FOLDER, exist_ok=True)  # Bikin folder hasil jika belum ada

# Load Model IndoBERT RoBERTa Sentiment Bahasa Indonesia
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="w11wo/indonesian-roberta-base-sentiment-classifier",
    tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier",
    truncation=True,
    max_length=512,
)


def bersihkan_teks(teks):
    # Emoji ke kata
    emoji_dict = {
        r"😂|🤣|😹": " lucu ",
        r"😊|😁|😄|🙂|😃|😆|😸|☺️": " senang ",
        r"😍|🥰|😘|😻": " cinta ",
        r"😢|😭|😞|😔|☹️|😿": " sedih ",
        r"😡|😠|🤬": " marah ",
        r"🤔|😕|😐|😑": " bingung ",
        r"😴|🥱|😪": " ngantuk ",
        r"👍|👌|✌️|👏": " bagus ",
        r"👎|🙄|😒": " buruk ",
    }
    for pattern, replacement in emoji_dict.items():
        teks = re.sub(pattern, replacement, teks)

    # Normalisasi kata gaul & slang
    slang_dict = {
        r"\bmantap\b|\bmantul\b|\bmantep\b|\bgokil\b": "bagus",
        r"\bzonk\b|\bampas\b|\bburikk\b": "buruk",
        r"\bbt\b|\bbete\b": "sedih",
        r"\bluar biasa\b": "bagus",
        r"\banjg\b|\bajg\b|\bkontol\b": "kasar",
        r"\bgpp\b": "gak apa-apa",
        r"\bmager\b": "malas gerak",
        r"\bngab\b": "teman",
        r"\bciyus\b": "serius",
        r"\bseblak\b": "makanan pedas",
        r"\bngopi\b": "minum kopi",
    }
    for slang, normal in slang_dict.items():
        teks = re.sub(slang, normal, teks)

    # Singkatan informal
    singkatan_dict = {
        r"\bgk\b": "gak",
        r"\bga\b": "gak",
        r"\btp\b": "tapi",
        r"\bdgn\b": "dengan",
        r"\bdr\b": "dari",
        r"\bdg\b": "dengan",
    }
    for sgk, normal in singkatan_dict.items():
        teks = re.sub(sgk, normal, teks)

    # Hilangkan URL, mention, hashtag, angka
    teks = re.sub(r"http\S+|www\S+", "", teks)  # url
    teks = re.sub(r"@\w+", "", teks)  # mention
    teks = re.sub(r"#\w+", "", teks)  # hashtag
    teks = re.sub(r"\d+", "", teks)  # angka

    # Hilangkan simbol spam
    teks = re.sub(r"!{2,}", "!", teks)
    teks = re.sub(r"\?{2,}", "?", teks)
    teks = re.sub(r"\s+", " ", teks)

    # Lowercase & trim
    teks = teks.lower().strip()
    return teks


# Konversi label + threshold score biar lebih akurat
def convert_label_with_threshold(label, score):
    if score < 0.6:
        return "Netral"
    elif label.lower() == "positive":
        return "Positif"
    elif label.lower() == "negative":
        return "Negatif"
    else:
        return "Netral"


# Fungsi batching aman untuk inference
def bert_batch_predict(text_list, batch_size=32):
    results = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        batch_results = sentiment_pipeline(batch)
        results.extend(batch_results)
    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/proses", methods=["POST"])
def proses():
    if "csvfile" not in request.files:
        flash("No file part", "error")
        return render_template("index.html")

    file = request.files["csvfile"]

    if file.filename == "":
        flash("No selected file", "error")
        return render_template("index.html")

    if not file.filename.endswith(".csv"):
        flash("Invalid file type. Please upload a CSV file.", "error")
        return render_template("index.html")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        flash(f"Error reading the CSV file: {e}", "error")
        return render_template("index.html")

    if "full_text" not in df.columns:
        flash("CSV file must contain a 'full_text' column", "error")
        return render_template("index.html")

    # Bersihkan teks komentar
    df["full_text"] = df["full_text"].astype(str).apply(bersihkan_teks)

    # Filter komentar kosong, NaN, atau angka tidak relevan
    df = df[
        df["full_text"].apply(
            lambda x: isinstance(x, str) and x.strip() != "" and not x.strip().isdigit()
        )
    ]
    df = df.reset_index(drop=True)

    # Kalau kosong setelah filter, tampilkan error
    if df.empty:
        flash("Tidak ada komentar valid setelah dibersihkan.", "error")
        return render_template("index.html")

    # Prediksi sentimen dengan batching
    sentiments = bert_batch_predict(df["full_text"].tolist(), batch_size=32)

    # Masukkan hasil ke dataframe
    df["sentiment_label_raw"] = [s["label"] for s in sentiments]
    df["sentiment_score"] = [s["score"] for s in sentiments]

    # Konversi label final dengan threshold
    df["sentiment"] = [
        convert_label_with_threshold(l, s)
        for l, s in zip(df["sentiment_label_raw"], df["sentiment_score"])
    ]

    # Simpan hasil CSV
    hasil_filename = "hasil_" + file.filename
    hasil_path = os.path.join(HASIL_FOLDER, hasil_filename)
    df.to_csv(hasil_path, index=False)

    return send_file(
        hasil_path,
        as_attachment=True,
        download_name=hasil_filename,
        mimetype="text/csv",
    )


@app.route("/analyze_csv", methods=["POST"])
def analyze_csv():
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file", "error")
        return redirect(url_for("index"))

    df = pd.read_csv(file)
    if "full_text" not in df.columns or "sentiment" not in df.columns:
        flash("CSV must contain 'full_text' and 'sentiment'", "error")
        return redirect(url_for("index"))

    distribusi_awal = df["sentiment"].value_counts()
    balance_before_split = "balance_before_split" in request.form

    if balance_before_split:
        # === Balancing sebelum split ===
        kelas = df["sentiment"].unique()
        max_count = df["sentiment"].value_counts().max()

        df_balanced = pd.concat(
            [
                resample(
                    df[df["sentiment"] == k],
                    replace=True,
                    n_samples=max_count,
                    random_state=42,
                )
                for k in kelas
            ]
        )

        distribusi_balanced = df_balanced["sentiment"].value_counts()

        X_train, X_test, y_train, y_test = train_test_split(
            df_balanced["full_text"],
            df_balanced["sentiment"],
            test_size=0.2,
            random_state=42,
            stratify=df_balanced["sentiment"],
        )

    else:
        # === Balancing setelah split ===
        X_train, X_test, y_train, y_test = train_test_split(
            df["full_text"],
            df["sentiment"],
            test_size=0.2,
            random_state=42,
            stratify=df["sentiment"],
        )

        train_df = pd.DataFrame({"full_text": X_train, "sentiment": y_train})
        kelas = train_df["sentiment"].unique()
        max_count = train_df["sentiment"].value_counts().max()

        train_df_balanced = pd.concat(
            [
                resample(
                    train_df[train_df["sentiment"] == k],
                    replace=True,
                    n_samples=max_count,
                    random_state=42,
                )
                for k in kelas
            ]
        )

        distribusi_balanced = train_df_balanced["sentiment"].value_counts()

        X_train = train_df_balanced["full_text"]
        y_train = train_df_balanced["sentiment"]

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

    hasil_df = pd.DataFrame(
        {"full_text": X_test, "actual": y_test, "predicted": y_pred}
    )
    hasil_df.to_csv(os.path.join(HASIL_FOLDER, "last_result.csv"), index=False)

    report_path = os.path.join(HASIL_FOLDER, "report.json")
    with open(report_path, "w") as f:
        import json

        json.dump(
            {
                "distribusi_awal": distribusi_awal.to_dict(),
                "distribusi_balanced": distribusi_balanced.to_dict(),
                "akurasi": akurasi,
                "report": report,
                "matrix": matrix.tolist(),
            },
            f,
        )

    joblib.dump(vectorizer, os.path.join(HASIL_FOLDER, "vectorizer.pkl"))
    joblib.dump(model, os.path.join(HASIL_FOLDER, "model.pkl"))

    return redirect(url_for("hasil", page=1))


@app.route("/hasil")
def hasil():
    hasil_path = os.path.join(HASIL_FOLDER, "last_result.csv")
    report_path = os.path.join(HASIL_FOLDER, "report.json")

    if not os.path.exists(hasil_path) or not os.path.exists(report_path):
        flash(
            "Belum ada hasil analisis. Silakan unggah dan analisis file terlebih dahulu.",
            "error",
        )
        return redirect(url_for("index"))

    # Baca data hasil dan metadata
    hasil_df = pd.read_csv(hasil_path)
    with open(report_path, "r") as f:
        meta = json.load(f)

    # Ambil filter dan page dari URL
    filter_value = request.args.get("filter", "").strip().lower()
    page = int(request.args.get("page", 1))
    per_page = 10

    # Filter berdasarkan kolom 'predicted' jika filter aktif
    if filter_value:
        hasil_df = hasil_df[
            hasil_df["predicted"].str.lower().str.contains(filter_value)
        ]

    # Hitung total data dan pagination
    total_data = len(hasil_df)
    total_pages = ceil(total_data / per_page) if total_data > 0 else 1
    start = (page - 1) * per_page
    end = start + per_page
    hasil_page = hasil_df.iloc[start:end].to_dict(orient="records")

    return render_template(
        "result.html",
        distribusi_awal=meta["distribusi_awal"],
        distribusi_balanced=meta["distribusi_balanced"],
        akurasi=meta["akurasi"],
        report=meta["report"],
        matrix=meta["matrix"],
        hasil=hasil_page,
        page=page,
        total_pages=total_pages,
        filter_value=filter_value,
    )


@app.route("/klasifikasi", methods=["POST"])
def klasifikasi():
    ulasan = request.form.get("ulasan")
    if not ulasan:
        return render_template("result.html", hasil_klasifikasi="(ulasan kosong)")

    vectorizer_path = os.path.join(HASIL_FOLDER, "vectorizer.pkl")
    model_path = os.path.join(HASIL_FOLDER, "model.pkl")
    hasil_path = os.path.join(HASIL_FOLDER, "last_result.csv")
    report_path = os.path.join(HASIL_FOLDER, "report.json")

    if not (
        os.path.exists(vectorizer_path)
        and os.path.exists(model_path)
        and os.path.exists(hasil_path)
        and os.path.exists(report_path)
    ):
        return render_template("result.html", hasil_klasifikasi="Model belum dilatih")

    import joblib, json

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    hasil_df = pd.read_csv(hasil_path)

    with open(report_path, "r") as f:
        meta = json.load(f)

    hasil = model.predict(vectorizer.transform([ulasan]))[0]

    page = 1
    per_page = 10
    hasil_page = hasil_df.iloc[0:per_page].to_dict(orient="records")

    return render_template(
        "result.html",
        hasil_klasifikasi=hasil,
        distribusi_awal=meta["distribusi_awal"],
        distribusi_balanced=meta["distribusi_balanced"],
        akurasi=meta["akurasi"],
        report=meta["report"],
        matrix=meta["matrix"],
        hasil=hasil_page,
        page=page,
        total_pages=ceil(len(hasil_df) / per_page),
    )


@app.route("/download_ulasan", methods=["POST"])
def download_ulasan():
    app_id = request.form.get("appid")
    count_str = request.form.get("count", "1000")

    if not app_id:
        flash("App ID kosong.", "error")
        return redirect(url_for("index"))

    try:
        count = int(count_str)
        if count > 10000:  # Batas maksimum agar tidak terlalu berat
            flash("Maksimum hanya bisa download 10000 ulasan.", "error")
            return redirect(url_for("index"))
    except ValueError:
        flash("Jumlah ulasan tidak valid.", "error")
        return redirect(url_for("index"))

    try:
        hasil_reviews = []
        next_token = None

        while len(hasil_reviews) < count:
            remaining = count - len(hasil_reviews)
            fetch_count = min(200, remaining)  # Play Store limit per request
            result, next_token = reviews(
                app_id,
                lang="id",
                country="id",
                sort=Sort.NEWEST,
                count=fetch_count,
                continuation_token=next_token,
            )
            hasil_reviews.extend(result)

            if not next_token:
                break  # Tidak ada lagi halaman selanjutnya

        if not hasil_reviews:
            flash("Tidak ada ulasan yang ditemukan.", "error")
            return redirect(url_for("index"))

        # Simpan ke DataFrame
        df = pd.DataFrame(hasil_reviews[:count])  # potong jika melebihi count

        df_simple = pd.DataFrame(
            {
                "userName": df["userName"],
                "score": df["score"],
                "at": df["at"],
                "full_text": df["content"],
            }
        )

        def categorize(score):
            if score >= 4:
                return "Positif"
            elif score == 3:
                return "Netral"
            else:
                return "Negatif"

        df_simple["kategori"] = df_simple["score"].apply(categorize)

        filename = f"ulasan_{app_id.replace('.', '_')}.csv"
        file_path = os.path.join(HASIL_FOLDER, filename)
        df_simple.to_csv(file_path, index=False)

        return send_file(
            file_path, as_attachment=True, download_name=filename, mimetype="text/csv"
        )

    except Exception as e:
        flash(f"Gagal mengambil ulasan: {e}", "error")
        return redirect(url_for("index"))


@app.route("/ambil_komentar_yt", methods=["POST"])
def ambil_komentar_yt():
    yt_url = request.form["yt_url"]
    jumlah_komentar = request.form.get("jumlah_komentar", type=int)

    video_id = extract_video_id(yt_url)
    if not video_id:
        return "Video ID tidak valid!", 400

    video_info = get_video_info(video_id)
    if not video_info:
        return "Video tidak ditemukan!", 404

    if not jumlah_komentar or jumlah_komentar <= 0:
        jumlah_komentar = 1000

    comments = get_comments(video_id, max_comments=jumlah_komentar)

    output_folder = "hasil"
    os.makedirs(output_folder, exist_ok=True)

    safe_title = "".join(
        c for c in video_info["video_title"] if c.isalnum() or c in (" ", "_")
    ).rstrip()
    filename = f"{safe_title}.csv"
    output_path = os.path.join(output_folder, filename)

    with open(output_path, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "video_title",
                "channel_title",
                "full_text",
                "author",
                "published_at",
                "like_count",
                "reply_count",
            ]
        )
        for comment in comments:
            writer.writerow(
                [
                    video_info["video_title"],
                    video_info["channel_title"],
                    comment["text"],
                    comment["author"],
                    comment["published_at"],
                    comment["like_count"],
                    comment["reply_count"],
                ]
            )

    return send_file(output_path, as_attachment=True, download_name=filename)


@app.route("/stream_yt", methods=["POST"])
def stream_yt():
    yt_url = request.form.get("yt_url")
    jumlah_komentar = int(request.form.get("jumlah_komentar", 1000))

    video_id = extract_video_id(yt_url)
    if not video_id:
        return Response("Video ID tidak valid!", mimetype="text/plain")

    video_info = get_video_info(video_id)
    if not video_info:
        return Response("Video tidak ditemukan!", mimetype="text/plain")

    def generate():
        comments = []
        next_token = None

        while len(comments) < jumlah_komentar:
            remaining = jumlah_komentar - len(comments)
            fetch_count = min(100, remaining)
            try:
                result, next_token = (
                    youtube.commentThreads()
                    .list(
                        part="snippet",
                        videoId=video_id,
                        textFormat="plainText",
                        maxResults=fetch_count,
                        pageToken=next_token,
                    )
                    .execute()
                )
            except Exception as e:
                yield f"\nERROR: {str(e)}\n"
                return

            for item in result.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comment_text = snippet["textDisplay"]
                comment_data = {
                    "text": comment_text,
                    "author": snippet.get("authorDisplayName", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "like_count": snippet.get("likeCount", 0),
                    "reply_count": item.get("snippet", {}).get("totalReplyCount", 0),
                }
                comments.append(comment_data)

            yield f"data: Mengambil komentar... ({len(comments)}/{jumlah_komentar})\n\n"
            time.sleep(0.3)

            if not next_token:
                break

        # Simpan ke file
        output_folder = "hasil"
        os.makedirs(output_folder, exist_ok=True)
        safe_title = "".join(
            c for c in video_info["video_title"] if c.isalnum() or c in (" ", "_")
        ).rstrip()
        filename = f"{safe_title}.csv"
        output_path = os.path.join(output_folder, filename)

        with open(output_path, mode="w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "video_title",
                    "channel_title",
                    "full_text",
                    "author",
                    "published_at",
                    "like_count",
                    "reply_count",
                ]
            )
            for comment in comments:
                writer.writerow(
                    [
                        video_info["video_title"],
                        video_info["channel_title"],
                        comment["text"],
                        comment["author"],
                        comment["published_at"],
                        comment["like_count"],
                        comment["reply_count"],
                    ]
                )

        yield f"data: SELESAI|{filename}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/analisis")
def analisis():
    return render_template("analisis.html")


@app.route("/panduan")
def panduan():
    return render_template("panduan.html")


@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")


@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_message = data.get("message", "").lower()

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
        "siapa kamu": "Saya adalah chatbot pendamping untuk aplikasi analisis sentimen berbasis Flask yang Anda gunakan.",
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
        response = responses.get(
            user_message,
            "Maaf, saya belum mengerti pertanyaan itu. Coba ketik 'menu' untuk melihat daftar bantuan.",
        )

    return {"response": response}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
