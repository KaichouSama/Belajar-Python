# import numpy as np
# import skfuzzy as fuzz
# from skfuzzy import control as ctrl

# # 1. Definisikan variabel input dan output
# frekuensi = ctrl.Antecedent(np.arange(0, 11, 1), 'frekuensi')
# belanja = ctrl.Antecedent(np.arange(0, 3000001, 10000), 'belanja')
# diskon = ctrl.Consequent(np.arange(0, 31, 1), 'diskon')

# # 2. Fungsi keanggotaan
# frekuensi['rendah'] = fuzz.trapmf(frekuensi.universe, [0, 0, 2, 4])
# frekuensi['sedang'] = fuzz.trapmf(frekuensi.universe, [3, 4, 6, 7])
# frekuensi['tinggi'] = fuzz.trapmf(frekuensi.universe, [6, 8, 10, 10])

# belanja['rendah'] = fuzz.trapmf(belanja.universe, [0, 0, 500000, 1000000])
# belanja['sedang'] = fuzz.trapmf(belanja.universe, [750000, 1250000, 1750000, 2250000])
# belanja['tinggi'] = fuzz.trapmf(belanja.universe, [2000000, 2500000, 3000000, 3000000])

# diskon['kecil'] = fuzz.trapmf(diskon.universe, [0, 0, 5, 10])
# diskon['sedang'] = fuzz.trapmf(diskon.universe, [8, 12, 18, 22])
# diskon['besar'] = fuzz.trapmf(diskon.universe, [20, 25, 30, 30])

# # Menampilkan grafik fungsi keanggotaan
# frekuensi.view()
# belanja.view()
# diskon.view()

# # 3. Aturan fuzzy
# rule1 = ctrl.Rule(frekuensi['rendah'] & belanja['rendah'], diskon['kecil'])
# rule2 = ctrl.Rule(frekuensi['rendah'] & belanja['sedang'], diskon['kecil'])
# rule3 = ctrl.Rule(frekuensi['rendah'] & belanja['tinggi'], diskon['sedang'])

# rule4 = ctrl.Rule(frekuensi['sedang'] & belanja['rendah'], diskon['kecil'])
# rule5 = ctrl.Rule(frekuensi['sedang'] & belanja['sedang'], diskon['sedang'])
# rule6 = ctrl.Rule(frekuensi['sedang'] & belanja['tinggi'], diskon['besar'])

# rule7 = ctrl.Rule(frekuensi['tinggi'] & belanja['rendah'], diskon['sedang'])
# rule8 = ctrl.Rule(frekuensi['tinggi'] & belanja['sedang'], diskon['besar'])
# rule9 = ctrl.Rule(frekuensi['tinggi'] & belanja['tinggi'], diskon['besar'])

# # 4. Sistem kontrol
# diskon_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
# diskon_simulasi = ctrl.ControlSystemSimulation(diskon_ctrl)

# # 5. Input dari pengguna
# try:
#     input_frekuensi = float(input("Masukkan frekuensi pembelian (0 - 10): "))
#     input_belanja = float(input("Masukkan total belanja (Rp, max 3000000): "))

#     # Validasi input
#     if not (0 <= input_frekuensi <= 10):
#         raise ValueError("Frekuensi harus antara 0 - 10")
#     if not (0 <= input_belanja <= 3000000):
#         raise ValueError("Belanja harus antara 0 - 3.000.000")

#     # 6. Simulasikan
#     diskon_simulasi.input['frekuensi'] = input_frekuensi
#     diskon_simulasi.input['belanja'] = input_belanja
#     diskon_simulasi.compute()

#     # 7. Output
#     print(f"\n🎁 Diskon yang direkomendasikan: {diskon_simulasi.output['diskon']:.2f}%")
# except ValueError as e:
#     print(f"Input error: {e}")

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Definisikan variabel input dan output
frekuensi = ctrl.Antecedent(np.arange(0, 11, 1), 'frekuensi')
belanja = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'belanja')  # dalam juta
diskon = ctrl.Consequent(np.arange(0, 31, 1), 'diskon')

# 2. Fungsi keanggotaan
frekuensi['rendah'] = fuzz.trapmf(frekuensi.universe, [0, 0, 2, 4])
frekuensi['sedang'] = fuzz.trapmf(frekuensi.universe, [3, 4, 6, 7])
frekuensi['tinggi'] = fuzz.trapmf(frekuensi.universe, [6, 8, 10, 10])

belanja['rendah'] = fuzz.trapmf(belanja.universe, [0, 0, 0.5, 1])
belanja['sedang'] = fuzz.trapmf(belanja.universe, [0.75, 1.25, 1.75, 2.25])
belanja['tinggi'] = fuzz.trapmf(belanja.universe, [2, 2.5, 3, 3])

diskon['kecil'] = fuzz.trapmf(diskon.universe, [0, 0, 5, 10])
diskon['sedang'] = fuzz.trapmf(diskon.universe, [8, 12, 18, 22])
diskon['besar'] = fuzz.trapmf(diskon.universe, [20, 25, 30, 30])

# 3. Tampilkan grafik fungsi keanggotaan (SEBELUM input user)
frekuensi.view()
belanja.view()
diskon.view()
plt.show()

# 4. Aturan fuzzy
rule1 = ctrl.Rule(frekuensi['rendah'] & belanja['rendah'], diskon['kecil'])
rule2 = ctrl.Rule(frekuensi['rendah'] & belanja['sedang'], diskon['kecil'])
rule3 = ctrl.Rule(frekuensi['rendah'] & belanja['tinggi'], diskon['sedang'])

rule4 = ctrl.Rule(frekuensi['sedang'] & belanja['rendah'], diskon['kecil'])
rule5 = ctrl.Rule(frekuensi['sedang'] & belanja['sedang'], diskon['sedang'])
rule6 = ctrl.Rule(frekuensi['sedang'] & belanja['tinggi'], diskon['besar'])

rule7 = ctrl.Rule(frekuensi['tinggi'] & belanja['rendah'], diskon['sedang'])
rule8 = ctrl.Rule(frekuensi['tinggi'] & belanja['sedang'], diskon['besar'])
rule9 = ctrl.Rule(frekuensi['tinggi'] & belanja['tinggi'], diskon['besar'])

# 5. Sistem kontrol
diskon_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
diskon_simulasi = ctrl.ControlSystemSimulation(diskon_ctrl)

# 6. Input dari pengguna
try:
    input_frekuensi = float(input("Masukkan frekuensi pembelian (0 - 10): "))
    input_belanja_rp = input("Masukkan total belanja (Rp, max 3.000.000): ").replace('.', '').replace(',', '')
    input_belanja = float(input_belanja_rp) / 1_000_000  # konversi ke juta

    if not (0 <= input_frekuensi <= 10):
        raise ValueError("Frekuensi harus antara 0 - 10")
    if not (0 <= input_belanja <= 3):
        raise ValueError("Belanja harus antara 0 - 3 juta")

    # 7. Simulasikan
    diskon_simulasi.input['frekuensi'] = input_frekuensi
    diskon_simulasi.input['belanja'] = input_belanja
    diskon_simulasi.compute()

    # 8. Output hasil
    print(f"\n🎁 Diskon yang direkomendasikan: {diskon_simulasi.output['diskon']:.2f}%")

except ValueError as e:
    print(f"Input error: {e}")
