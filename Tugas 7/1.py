import numpy as np
import pandas as pd

# Data input dan label
X = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 1],
    [0, 1, 0]
])
y = np.array([[1], [0], [0], [1], [1], [0], [1]])

# Fungsi aktivasi dan turunannya
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1 - np.tanh(x)**2

def linear(x): return x
def linear_derivative(x): return np.ones_like(x)

# Fungsi training jaringan
def train(X, y, activation, activation_deriv, epochs=2500, lr=0.05):
    np.random.seed(1)
    weights = 2 * np.random.random((3, 1)) - 1
    bias = np.random.rand()
    for _ in range(epochs):
        z = np.dot(X, weights) + bias
        output = activation(z)
        error = y - output
        adjustments = lr * np.dot(X.T, error * activation_deriv(output))
        weights += adjustments
        bias += lr * np.sum(error * activation_deriv(output))
    return weights, bias

# Fungsi prediksi
def predict(X, weights, bias, activation):
    return activation(np.dot(X, weights) + bias)

# 1 & 2. Perbandingan fungsi aktivasi
activations = {
    'Sigmoid': (sigmoid, sigmoid_derivative),
    'Tanh': (tanh, tanh_derivative),
    'Linear': (linear, linear_derivative)
}

results = {}
for name, (act, d_act) in activations.items():
    w, b = train(X, y, act, d_act)
    pred1 = predict(np.array([[1, 0, 0]]), w, b, act)[0][0]
    pred2 = predict(np.array([[0, 1, 0]]), w, b, act)[0][0]
    results[name] = [round(pred1, 4), round(pred2, 4)]

# Tabel hasil prediksi per fungsi aktivasi
activation_comparison = pd.DataFrame(results, index=["[1, 0, 0]", "[0, 1, 0]"]).T
print("Perbandingan Hasil Prediksi Berdasarkan Fungsi Aktivasi:\n")
print(activation_comparison)

# 3. Perbandingan prediksi berdasarkan variasi epoch (menggunakan sigmoid)
epoch_results = {}
for epoch in [2500, 3500, 4500]:
    w, b = train(X, y, sigmoid, sigmoid_derivative, epochs=epoch)
    pred1 = predict(np.array([[1, 0, 0]]), w, b, sigmoid)[0][0]
    pred2 = predict(np.array([[0, 1, 0]]), w, b, sigmoid)[0][0]
    epoch_results[epoch] = [round(pred1, 4), round(pred2, 4)]

# Tabel hasil prediksi per epoch
epoch_comparison = pd.DataFrame(epoch_results, index=["[1, 0, 0]", "[0, 1, 0]"]).T
print("\nPerbandingan Hasil Prediksi Berdasarkan Variasi Epoch:\n")
print(epoch_comparison)
