# problema7_tp2.py
import pandas as pd
import numpy as np
from scipy.stats import norm, t

# leer datos
ruta_csv = r"RYR1.csv"
df = pd.read_csv(ruta_csv)

# proporciones de genotipos
conteos = df['Genotipo'].value_counts()
total = conteos.sum()
print("Proporciones de genotipos (con IC 95%):")
for genotipo in ['CC', 'CT', 'TT']:
    p = conteos[genotipo] / total
    error = np.sqrt(p * (1 - p) / total)
    z = norm.ppf(1 - 0.05 / 2)
    li = p - z * error
    ls = p + z * error
    print(f"{genotipo}: {p:.4f} ({li:.4f}, {ls:.4f})")

# precisión de los IC
print("\nPrecisión de cada intervalo:")
for genotipo in ['CC', 'CT', 'TT']:
    p = conteos[genotipo] / total
    error = np.sqrt(p * (1 - p) / total)
    print(f"{genotipo}: ±{(z * error):.4f}")

# tamaño muestral para estimar TT con ±1%
p_tt = conteos['TT'] / total
margen_error = 0.01
n_necesario = (z ** 2 * p_tt * (1 - p_tt)) / (margen_error ** 2)
print(f"\nTamaño muestral necesario para estimar % de TT con ±1%: {int(np.ceil(n_necesario))}")

# IC para pH por genotipo
print("\nIntervalos de confianza para pH promedio (95%):")
for genotipo in ['CC', 'CT', 'TT']:
    subset = df[df['Genotipo'] == genotipo]['pH_canal'].dropna()
    media = np.mean(subset)
    desvio = np.std(subset, ddof=1)
    n = len(subset)
    error = desvio / np.sqrt(n)
    t_crit = t.ppf(1 - 0.05 / 2, df=n - 1)
    li = media - t_crit * error
    ls = media + t_crit * error
    print(f"{genotipo}: {media:.4f} ({li:.4f}, {ls:.4f})")

# tamaño muestral para duplicar precisión en TT
subset_tt = df[df['Genotipo'] == 'TT']['pH_canal'].dropna()
n_tt = len(subset_tt)
n_doble_precision = n_tt * 4
print(f"\nTamaño muestral necesario para duplicar la precisión en pH de TT: {n_doble_precision}")

print("\nParámetros estimados:")
print("- % de CC, CT, TT (ítem 1)")
print("- Media de pH por genotipo (ítem 4)")
print("Población: cerdos faenados en el frigorífico La Pompeya.")
