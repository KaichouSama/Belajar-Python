import pandas as pd
from transformers import pipeline

# Load file CSV
df = pd.read_csv("C:/Users/zeroo/OneDrive/Documents/Project Mandiri/AI (Python)/Hugging Face/prabowo1.csv")

# Buat pipeline analisis sentimen dengan model multilingual
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Lakukan analisis sentimen pada kolom full_text
sentiments = sentiment_pipeline(df["full_text"].astype(str).tolist())

# Konversi label dari 1-5 stars ke sentimen positif/netral/negatif
def convert_label(label):
    if label in ['1 star', '2 stars']:
        return 'Negatif'
    elif label == '3 stars':
        return 'Netral'
    else:
        return 'Positif'

# Tambahkan hasil ke DataFrame
df["sentiment_label_raw"] = [s["label"] for s in sentiments]
df["sentiment_score"] = [s["score"] for s in sentiments]
df["sentiment"] = df["sentiment_label_raw"].apply(convert_label)

# Simpan hasil ke file baru
df.to_csv("hasil_sentimen_prabowo.csv", index=False)

print("Analisis sentimen selesai! Hasil disimpan di 'hasil_sentimen_prabowo.csv'")
