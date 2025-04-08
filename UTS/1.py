# import csv
# import math
# from collections import defaultdict

# # 1. Load dataset
# def load_dataset(filename):
#     with open(filename, 'r') as file:
#         reader = csv.reader(file)
#         dataset = list(reader)
#         header = dataset.pop(0)  # Buang header
#         for row in dataset:
#             for i in range(8):
#                 row[i] = float(row[i])  # Konversi fitur ke float
#         return dataset

# # 2. Pisahkan berdasarkan kelas
# def separate_by_class(dataset):
#     separated = defaultdict(list)
#     for row in dataset:
#         label = row[-1]
#         separated[label].append(row[:-1])  # Tanpa label
#     return separated

# # 3. Hitung mean dan standar deviasi
# def mean(numbers):
#     return sum(numbers) / float(len(numbers))

# def stdev(numbers):
#     avg = mean(numbers)
#     variance = sum([(x - avg)**2 for x in numbers]) / float(len(numbers))
#     return math.sqrt(variance)

# def summarize_dataset(dataset):
#     summaries = [(mean(column), stdev(column)) for column in zip(*dataset)]
#     return summaries

# def summarize_by_class(dataset):
#     separated = separate_by_class(dataset)
#     summaries = {}
#     for class_value, rows in separated.items():
#         summaries[class_value] = summarize_dataset(rows)
#     return summaries

# # 4. Hitung probabilitas
# def calculate_probability(x, mean, stdev):
#     if stdev == 0:
#         return 1 if x == mean else 0
#     exponent = math.exp(-(x - mean)**2 / (2 * stdev**2))
#     return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# def calculate_class_probabilities(summaries, input_vector):
#     probabilities = {}
#     for class_value, class_summaries in summaries.items():
#         probabilities[class_value] = 1
#         for i in range(len(class_summaries)):
#             mean_i, stdev_i = class_summaries[i]
#             x = input_vector[i]
#             probabilities[class_value] *= calculate_probability(x, mean_i, stdev_i)
#     return probabilities

# # 5. Prediksi
# def predict(summaries, input_vector):
#     probabilities = calculate_class_probabilities(summaries, input_vector)
#     best_label, best_prob = None, -1
#     for class_value, probability in probabilities.items():
#         if best_label is None or probability > best_prob:
#             best_label, best_prob = class_value, probability
#     return best_label, probabilities

# # 6. Jalankan program
# filename = 'C:/Users/zeroo/OneDrive/Documents/Project Mandiri/AI (Python)/UTS/breast-cancer.csv'
# dataset = load_dataset(filename)
# summaries = summarize_by_class(dataset)

# # 7. Data pasien baru
# new_patient = [14.5, 20.3, 70.6, 500.0, 0.1, 0.15, 0.2, 0.3]
# predicted_class, probs = predict(summaries, new_patient)

# # 8. Output
# print("Prediksi kelas:", "Malignant (Ganas)" if predicted_class == 'M' else "Benign (Jinak')")
# print("Probabilitas:")
# for k, v in probs.items():
#     label = "Malignant" if k == 'M' else "Benign"
#     print(f"  {label}: {v:.8f}")

import pandas as pd
from sklearn.naive_bayes import GaussianNB

# 1. Load dataset
filename = 'C:/Users/zeroo/OneDrive/Documents/Project Mandiri/AI (Python)/UTS/breast-cancer.csv'
df = pd.read_csv(filename)

# 2. Pisahkan fitur dan label
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# 3. Latih model Naive Bayes
model = GaussianNB()
model.fit(X, y)

# 4. Data pasien baru (PASTIKAN pakai DataFrame dan kolom yang sama!)
new_data = pd.DataFrame([{
    'radius_mean': 14.5,
    'texture_mean': 20.3,
    'perimeter_mean': 70.6,
    'area_mean': 500.0,
    'smoothness_mean': 0.1,
    'compactness_mean': 0.15,
    'concavity_mean': 0.2,
    'symmetry_mean': 0.3
}])

# 5. Prediksi
predicted_class = model.predict(new_data)[0]
probs = model.predict_proba(new_data)[0]

# 6. Output hasil
print("Prediksi kelas:", "Malignant (Ganas)" if predicted_class == 'M' else "Benign (Jinak)")
print("Probabilitas:")
for label, prob in zip(model.classes_, probs):
    label_full = "Malignant" if label == 'M' else "Benign"
    print(f"  {label_full}: {prob:.8f}")
