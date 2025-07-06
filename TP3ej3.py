import numpy as np
from scipy import stats
import pandas as pd

## Problema 3 - Emergencia sanitaria
n = 300
x = 300 - 165  # muestras contaminadas
prop_critica = 0.40

# Prueba de proporciones unilateral
print("\nProblema 3 - Emergencia sanitaria:")
stat, pval = stats.binomtest(x, n, prop_critica, alternative='greater')
print(f"Proporción observada: {x/n:.4f}, p-valor: {pval:.4f}")
if pval < 0.05:
    print("Conclusión: Rechazamos H0 - Hay evidencia para declarar emergencia (p < 0.05)")
else:
    print("Conclusión: No hay evidencia suficiente para declarar emergencia (p >= 0.05)")

# Intervalo de confianza 90%
ci_low, ci_upp = stats.proportion_confint(x, n, alpha=0.10, method='wilson')
print(f"Intervalo de confianza 90%: [{ci_low:.4f}, {ci_upp:.4f}]")
