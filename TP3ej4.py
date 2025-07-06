import numpy as np
from scipy import stats
import pandas as pd

## Problema 4 - Análisis genotipo RYR1
# Cargar datos
datos = pd.read_csv("RYR1.csv")
datos_filtrados = datos[datos['Genotipo'].isin(['TT', 'CT'])]

# 1. Prueba t unilateral izquierda para pH
print("\nProblema 4 - Ítem 1 (pH):")
t_ph, p_ph = stats.ttest_1samp(datos_filtrados['pH'], popmean=6.2, alternative='less')
print(f"Estadístico t: {t_ph:.4f}, p-valor: {p_ph:.4f}")
if p_ph < 0.05:
    print("Conclusión: Rechazamos H0 - El pH medio es menor que 6.2 (p < 0.05)")
else:
    print("Conclusión: No hay evidencia de pH menor a 6.2 (p >= 0.05)")

# 2. Prueba de proporciones
n_tt = sum(datos['Genotipo'] == 'TT')
n_total = len(datos)
print("\nProblema 4 - Ítem 2 (Proporción TT):")
stat, pval = stats.binomtest(n_tt, n_total, p=0.05, alternative='greater')
print(f"Proporción observada TT: {n_tt/n_total:.4f}, p-valor: {pval:.4f}")
if pval < 0.05:
    print("Conclusión: Rechazamos H0 - La proporción de TT supera el 5% (p < 0.05)")
else:
    print("Conclusión: No hay evidencia de proporción mayor al 5% (p >= 0.05)")


