from flask import Flask, render_template, send_file
import pandas as pd
from google_play_scraper import reviews, Sort

app = Flask(__name__)

# ====== STEP 1: SCRAPING REVIEW ======
def get_reviews():
    result, _ = reviews(
        'com.shopee.id',  # ID aplikasi Shopee
        lang='id',        # Bahasa Indonesia
        country='id',     # Negara Indonesia
        sort=Sort.NEWEST, # Mengambil ulasan terbaru
        count=1000        # Mengambil 1000 ulasan
    )

    # Membuat DataFrame dari hasil scraping
    df = pd.DataFrame(result)[['content', 'score']]

    # Labeling kategori berdasarkan score
    def get_sentiment(score):
        if score >= 4:
            return "Positif"
        elif score <= 2:
            return "Negatif"
        else:
            return "Netral"

    df['kategori'] = df['score'].apply(get_sentiment)
    df = df[df['kategori'] != 'Netral']  # Mengambil hanya Positif & Negatif
    df = df[['content', 'kategori']]
    df.columns = ['full_text', 'kategori']

    return df

# Data ulasan dari Google Play
df = get_reviews()

# Route utama untuk menampilkan halaman
@app.route('/')
def index():
    return render_template('index.html', data=df.to_dict(orient='records'))

# Route untuk mengunduh CSV
@app.route('/download_csv')
def download_csv():
    # Menyimpan DataFrame ke file CSV
    csv_file = 'google_play_reviews.csv'
    df.to_csv(csv_file, index=False)
    
    # Kirim file CSV untuk diunduh
    return send_file(csv_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
