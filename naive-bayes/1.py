import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample

# 1. Load Data
df = pd.read_csv("C:/Users/zeroo/OneDrive/Documents/Project Mandiri/AI (Python)/Project Akhir/hasil_sentimen_prabowo.csv")
df = df[["full_text", "sentiment"]].dropna()
df["full_text"] = df["full_text"].astype(str)

# 2. Cek distribusi awal (opsional)
print("Distribusi awal:\n", df["sentiment"].value_counts())

# 3. Balancing Dataset dengan Oversampling
negatif = df[df['sentiment'] == 'Negatif']
netral = df[df['sentiment'] == 'Netral']
positif = df[df['sentiment'] == 'Positif']

max_class_count = max(len(negatif), len(netral), len(positif))

netral_upsampled = resample(netral, replace=True, n_samples=max_class_count, random_state=42)
positif_upsampled = resample(positif, replace=True, n_samples=max_class_count, random_state=42)
negatif_upsampled = resample(negatif, replace=True, n_samples=max_class_count, random_state=42)

df_balanced = pd.concat([negatif_upsampled, netral_upsampled, positif_upsampled])
print("\nDistribusi setelah balancing:\n", df_balanced["sentiment"].value_counts())

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["full_text"], df_balanced["sentiment"], test_size=0.2, random_state=42
)

# 5. TF-IDF dengan n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Pilih dan Latih Model
# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)

print("\n=== Naive Bayes ===")
print("Akurasi:", accuracy_score(y_test, nb_pred))
print("Classification Report:\n", classification_report(y_test, nb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_pred))

# Logistic Regression (perbandingan)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)

print("\n=== Logistic Regression ===")
print("Akurasi:", accuracy_score(y_test, lr_pred))
print("Classification Report:\n", classification_report(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
