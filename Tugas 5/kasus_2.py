import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Membaca CSV dengan nama kolom yang sesuai
df = pd.read_csv("C:/Users/zeroo/OneDrive/Documents/Project Mandiri/AI (Python)/Tugas 5/pembeli.csv", names=["Umur", "Pendapatan", "Beli"])

# Menampilkan data awal
print("Data Awal:")
print(df.head())

# Memisahkan fitur (X) dan target (y)
X = df[["Umur", "Pendapatan"]]  # Menggunakan dua kolom pertama sebagai fitur
y = df["Beli"]  # Menggunakan kolom terakhir sebagai target

# Membagi dataset menjadi data latih (train) dan data uji (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Naïve Bayes
model = GaussianNB()

# Melatih model dengan data latih
model.fit(X_train, y_train)

# Memprediksi data uji
y_pred = model.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Memprediksi satu sample baru
sample_input = [[44, 6089827]]  # Hanya dua fitur: Umur dan Pendapatan
predicted_class = model.predict(sample_input)
print("Prediksi kelas untuk sample", sample_input, "adalah:", predicted_class[0])
