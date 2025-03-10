from csv import reader
from math import sqrt, exp, pi

# Fungsi untuk membaca dataset dengan mengabaikan header
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        next(csv_reader)  # **Lewati header**
        for row in csv_reader:
            cleaned_row = [float(value) if value.replace('.', '', 1).isdigit() else value.strip('"') for value in row]
            dataset.append(cleaned_row)
    return dataset

# Konversi nilai fitur ke float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column])

# Konversi kelas menjadi angka (label encoding)
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = {value: idx for idx, value in enumerate(unique)}
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Pisahkan dataset berdasarkan kelasnya
def separate_by_class(dataset):
    separated = {}
    for row in dataset:
        class_value = row[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(row)
    return separated

# Hitung mean dan standar deviasi
def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / (len(numbers) - 1)
    return sqrt(variance)

# Meringkas dataset (mean dan standar deviasi per fitur)
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column)) for column in zip(*dataset)][:-1]  # Tanpa kelas
    return summaries

# Meringkas dataset berdasarkan kelas
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {class_value: summarize_dataset(rows) for class_value, rows in separated.items()}
    return summaries

# Fungsi Gaussian untuk menghitung probabilitas
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Menghitung probabilitas setiap kelas untuk sebuah data
def calculate_class_probabilities(summaries, row):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

# Prediksi kelas berdasarkan probabilitas tertinggi
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    return max(probabilities, key=probabilities.get)

# Program utama
filename = "C:/Users/zeroo/OneDrive/Documents/Project Mandiri/AI (Python)/Tugas 3/iris.csv"
dataset = load_csv(filename)

# Konversi fitur ke float
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)

# Konversi kolom kelas menjadi angka
lookup = str_column_to_int(dataset, len(dataset[0])-1)

# Latih model Naïve Bayes
model = summarize_by_class(dataset)

# Data baru yang akan diprediksi
row = [5.7, 2.9, 4.2, 1.3]

# Prediksi label
label = predict(model, row)

# Cetak hasil prediksi
print(f'Data={row}, Predicted Class Index: {label}')

# Menampilkan nama kelas aslinya
for key, value in lookup.items():
    if value == label:
        print(f'Predicted Class Name: {key}')
