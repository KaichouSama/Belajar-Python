import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import pandas as pd
import matplotlib.pyplot as plt

# Membaca dataset employee.csv (harus tersedia dalam direktori kerja)
df = pd.read_csv("C:/Users/zeroo/OneDrive/Documents/Project Mandiri/AI (Python)/Tugas 4/employee.csv")

# Menghapus spasi di nama kolom agar lebih mudah digunakan
df.columns = df.columns.str.replace(" ", "_")

# Definisi Variabel Fuzzy
age = ctrl.Antecedent(np.arange(20, 61, 1), 'age')
years_of_service = ctrl.Antecedent(np.arange(0, 41, 1), 'years_of_service')
salary = ctrl.Antecedent(np.arange(2000, 7001, 100), 'salary')
status = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'status')

# Definisi Fungsi Keanggotaan Trapesium dan Gaussian
age['young'] = fuzz.trapmf(age.universe, [20, 20, 30, 35])
age['middle'] = fuzz.gaussmf(age.universe, 40, 5)
age['old'] = fuzz.trapmf(age.universe, [45, 50, 60, 60])

years_of_service['short'] = fuzz.trapmf(years_of_service.universe, [0, 0, 5, 10])
years_of_service['medium'] = fuzz.gaussmf(years_of_service.universe, 20, 5)
years_of_service['long'] = fuzz.trapmf(years_of_service.universe, [30, 35, 40, 40])

salary['low'] = fuzz.trapmf(salary.universe, [2000, 2000, 3000, 4000])
salary['medium'] = fuzz.gaussmf(salary.universe, 5000, 800)
salary['high'] = fuzz.trapmf(salary.universe, [5500, 6000, 7000, 7000])

# Definisi Fungsi Keanggotaan untuk Status
status['contract'] = fuzz.trimf(status.universe, [0, 0, 0.5])  # Karyawan kontrak
status['permanent'] = fuzz.trimf(status.universe, [0.5, 1, 1])  # Karyawan tetap

# Definisi Rules
rule1 = ctrl.Rule(age['old'] & years_of_service['long'] & salary['high'], status['permanent'])
rule2 = ctrl.Rule(age['young'] & years_of_service['short'] & salary['low'], status['contract'])
rule3 = ctrl.Rule(age['middle'] & years_of_service['medium'] & salary['medium'], status['permanent'])

# Pembuatan Sistem Fuzzy
status_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
status_simulation = ctrl.ControlSystemSimulation(status_ctrl)

# Evaluasi Data
results = []
for index, row in df.iterrows():
    status_simulation.input['age'] = row['Age']
    status_simulation.input['years_of_service'] = row['Years_of_Service']
    status_simulation.input['salary'] = row['Salary']
    status_simulation.compute()
    results.append((row['Name'], "Tetap" if status_simulation.output['status'] > 0.5 else "Kontrak"))

# Menampilkan hasil
for name, stat in results:
    print(f"Karyawan: {name}, Status: {stat}")

# Visualisasi
age.view()
years_of_service.view()
salary.view()
status.view()
plt.show()
