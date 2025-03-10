import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randint(0, 100, size=(3, 4)), columns=['Zahra', 'Lily', 'Dito', 'Naufal'])

print("Seluruh DataFrame:")
print(df)

print("\nNilai dalam sel baris #1 dari kolom Lily, Naufal, dan Zahra:")
print("Kelas A (Lily):", df.loc[0, 'Lily'])
print("Kelas B (Naufal):", df.loc[0, 'Naufal'])
print("Kelas C (Zahra):", df.loc[0, 'Zahra'])

df['Janet'] = df['Zahra'] + df['Dito']

print("\nDataFrame yang telah diperbarui:")
print(df)
