import numpy as np
from scipy import stats
import pandas as pd

## Problema 2 - Contenido de calcio
calcio_A234 = np.array([
    252.49, 248.16, 251.65, 249.04, 246.85, 251.16,
    247.98, 248.29, 246.86, 253.11, 250.84, 252.74,
    252.26, 250.16, 250.98, 247.63, 250.93, 250.53,
    250.44, 249.22
])

calcio_G120 = np.array([
    248.48, 249.88, 250.94, 245.61, 247.33, 247.32,
    253.52, 247.87, 248.03, 248.42, 244.68, 250.51,
    246.45, 250.06, 245.49, 247.15, 254.03, 246.36,
    250.06, 249.29
])

# Pruebas t bilaterales
print("\nProblema 2 - Lote A234:")
t_A234, p_A234 = stats.ttest_1samp(calcio_A234, popmean=250)
print(f"Estadístico t: {t_A234:.4f}, p-valor: {p_A234:.4f}")
if p_A234 < 0.01:
    print("Conclusión: Rechazamos H0 - El proceso debe detenerse (p < 0.01)")
else:
    print("Conclusión: No se rechaza H0 - El proceso puede continuar (p >= 0.01)")

print("\nProblema 2 - Lote G120:")
t_G120, p_G120 = stats.ttest_1samp(calcio_G120, popmean=250)
print(f"Estadístico t: {t_G120:.4f}, p-valor: {p_G120:.4f}")
if p_G120 < 0.01:
    print("Conclusión: Rechazamos H0 - El proceso debe detenerse (p < 0.01)")
else:
    print("Conclusión: No se rechaza H0 - El proceso puede continuar (p >= 0.01)")

