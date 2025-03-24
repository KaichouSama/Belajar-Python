# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, classification_report

# # Membaca dataset
# file_path = "/mnt/data/pembeli.csv"
# df = pd.read_csv(file_path, header=None, names=["Usia", "Pendapatan", "Membeli"])

# # Memisahkan fitur (X) dan label (y)
# X = df[["Usia", "Pendapatan"]]
# y = df["Membeli"]

# # Membagi data menjadi training (80%) dan testing (20%)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Inisialisasi model Naïve Bayes
# model = GaussianNB()

# # Melatih model
# model.fit(X_train, y_train)

# # Memprediksi data uji
# y_pred = model.predict(X_test)

# # Evaluasi model
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print(f"Akurasi Model: {accuracy:.2f}")
# print("Laporan Klasifikasi:\n", report)
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, classification_report

# # Membaca dataset
# file_path = "/mnt/data/pembeli.csv"
# df = pd.read_csv(file_path, header=None, names=["Usia", "Pendapatan", "Membeli"])

# # Memisahkan fitur (X) dan label (y)
# X = df[["Usia", "Pendapatan"]]
# y = df["Membeli"]

# # Membagi data menjadi training (80%) dan testing (20%)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Inisialisasi model Naïve Bayes
# model = GaussianNB()

# # Melatih model
# model.fit(X_train, y_train)

# # Memprediksi data uji
# y_pred = model.predict(X_test)

# # Evaluasi model
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print(f"Akurasi Model: {accuracy:.2f}")
# print("Laporan Klasifikasi:\n", report)
