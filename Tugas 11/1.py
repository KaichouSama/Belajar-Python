import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("c:/Users/zeroo/OneDrive/Documents/Project Mandiri/AI (Python)/Tugas 11/happydata.csv")

# Bersihkan nama kolom (hilangkan karakter aneh seperti ë)
df.columns = df.columns.str.strip().str.lower().str.replace("ë", "e")

# Tampilkan ringkasan awal
print(df.head())
print(df.info())
print(df['happy'].value_counts())

# Encode target jika perlu (opsional, sudah 0 dan 1 sih)
le = LabelEncoder()
df['happy'] = le.fit_transform(df['happy'])

# Cek null
print(df.isnull().sum())

# Pisahkan fitur dan target
X = df.drop('happy', axis=1)
y = df['happy']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Daftar model
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Evaluasi model
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\nModel: {name}")
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualisasi akurasi
plt.figure(figsize=(10, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.ylabel("Accuracy")
plt.title("Perbandingan Akurasi Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
