import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
# Variabel input (Kelembaban)
kelembapan = ctrl.Antecedent(np.arange(0, 101, 1), 'kelembapan')

# Variabel output (Waktu Penyiraman)
waktu_siram = ctrl.Consequent(np.arange(0, 31, 1), 'waktu_siram')

# Fungsi keanggotaan untuk kelembapan
kelembapan['kering'] = fuzz.trimf(kelembapan.universe, [0, 0, 30])
kelembapan['lembab'] = fuzz.trimf(kelembapan.universe, [20, 50, 60])
kelembapan['basah'] = fuzz.trimf(kelembapan.universe, [50, 100, 100])

# Fungsi keanggotaan untuk waktu penyiraman
waktu_siram['cepat'] = fuzz.trimf(waktu_siram.universe, [0, 0, 10])
waktu_siram['sedang'] = fuzz.trimf(waktu_siram.universe, [5, 15, 20])
waktu_siram['lambat'] = fuzz.trimf(waktu_siram.universe, [15, 30, 30])

rule1 = ctrl.Rule(kelembapan['kering'], waktu_siram['cepat'])
rule2 = ctrl.Rule(kelembapan['lembab'], waktu_siram['sedang'])
rule3 = ctrl.Rule(kelembapan['basah'], waktu_siram['lambat'])

siram_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
siram_simulasi = ctrl.ControlSystemSimulation(siram_ctrl)

siram_simulasi.input['kelembapan'] = 45
siram_simulasi.compute()

print(f"Waktu Penyiraman: {siram_simulasi.output['waktu_siram']:.2f} menit")

kelembapan.view()
waktu_siram.view()
plt.show()
