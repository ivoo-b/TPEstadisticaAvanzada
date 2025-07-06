

# Prueba de bondad de ajuste - Problema 1 TP4

import scipy.stats as stats

# Frecuencias observadas
observados = [85, 28, 35, 12]

# Probabilidades esperadas bajo el modelo genético 9:3:3:1
probabilidades_esperadas = [9/16, 3/16, 3/16, 1/16]

# Prueba de chi-cuadrado
chi2, p = stats.chisquare(f_obs=observados, f_exp=[sum(observados)*p for p in probabilidades_esperadas])

print("Estadístico chi-cuadrado:", chi2)
print("Valor p:", p)

if p < 0.05:
    print("Se rechaza H0: La distribución observada se aparta del modelo genético 9:3:3:1.")
else:
    print("No se rechaza H0: No hay evidencia suficiente para decir que se aparta del modelo genético.")
