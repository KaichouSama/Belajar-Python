# Install dulu jika belum:
# pip install google-play-scraper pandas scikit-learn Sastrawi openpyxl

from google_play_scraper import reviews, Sort
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

# ====== STEP 1: SCRAPING REVIEW ======
print("📥 Mengambil data ulasan Shopee dari Google Play...")
result, _ = reviews(
    'com.shopee.id',
    lang='id',
    country='id',
    sort=Sort.NEWEST,
    count=1000
)

# Buat DataFrame
df = pd.DataFrame(result)[['content', 'score']]

# Labeling kategori
def get_sentiment(score):
    if score >= 4:
        return "Positif"
    elif score <= 2:
        return "Negatif"
    else:
        return "Netral"

df['kategori'] = df['score'].apply(get_sentiment)
df = df[df['kategori'] != 'Netral']  # Kita hanya ambil Positif & Negatif
df = df[['content', 'kategori']]
df.columns = ['ulasan', 'kategori']

# ====== STEP 2: TEXT PREPROCESSING ======
print("🧹 Melakukan preprocessing teks...")

stop_factory = StopWordRemoverFactory()
stop_words = set(stop_factory.get_stop_words())

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def preprocessing(text):
    text = text.lower()  # Case folding
    text = re.sub(r'[^a-z\s]', '', text)  # Remove symbols/angka
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    return ' '.join(stemmed)

df['clean'] = df['ulasan'].apply(preprocessing)

# ====== STEP 3: TF-IDF ======
print("🔠 Mengubah teks ke TF-IDF vector...")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean'])
y = df['kategori']

# ====== STEP 4: SPLITTING DAN KLASIFIKASI ======
print("🧠 Melatih model Naive Bayes...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# ====== STEP 5: EVALUASI MODEL ======
y_pred = model.predict(X_test)
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

# ====== (Optional) Simpan data ======
df.to_csv("shopee_reviews_clean.csv", index=False)
print("\n📁 Data dan hasil telah disimpan ke 'shopee_reviews_clean.csv'")
