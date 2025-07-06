import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import ttest_ind

# Cargar los datos
diatraea = pd.read_csv('Diatraea.csv')

# 1. Análisis de resistencia al ataque

print("1. ANÁLISIS DE RESISTENCIA AL ATAQUE")
print("-----------------------------------")

# Datos proporcionados en el enunciado
n_Bt = 50
danadas_Bt = 7
n_noBt = 50
danadas_noBt = 15

# Proporciones
p_Bt = danadas_Bt / n_Bt
p_noBt = danadas_noBt / n_noBt

print(f"\nProporción de plantas dañadas:")
print(f"- Híbrido Bt: {p_Bt:.3f} ({danadas_Bt}/{n_Bt})")
print(f"- Híbrido no Bt: {p_noBt:.3f} ({danadas_noBt}/{n_noBt})")

# Prueba de hipótesis para dos proporciones
print("\nPrueba de hipótesis:")
# H0: p_Bt = p_noBt (no hay diferencia en resistencia)
# H1: p_Bt < p_noBt (el Bt es más resistente)

count = np.array([danadas_Bt, danadas_noBt])
nobs = np.array([n_Bt, n_noBt])
zstat, pval = proportions_ztest(count, nobs, alternative='smaller')

print(f"Estadístico z = {zstat:.4f}")
print(f"Valor p = {pval:.5f}")

# Tamaño del efecto (diferencia de riesgo)
risk_diff = p_Bt - p_noBt
print(f"\nDiferencia de riesgo: {risk_diff:.3f}")

# Riesgo relativo
rr = p_Bt / p_noBt
print(f"Riesgo relativo: {rr:.3f}")

# 2. Análisis de rendimiento (peso de mazorca)

print("\n\n2. ANÁLISIS DE RENDIMIENTO")
print("--------------------------")

# Asumiremos que los datos en el CSV corresponden a:
# - Las primeras 9 filas son Bt (parcelas 1-9 con % infestación 0-12%)
# - Las últimas 9 filas son no Bt (parcelas 10-18 con % infestación 18-30%)

peso_Bt = diatraea['peso'][:9]
peso_noBt = diatraea['peso'][9:]

print("\nEstadísticos descriptivos:")
print("Híbrido Bt:")
print(peso_Bt.describe())
print("\nHíbrido no Bt:")
print(peso_noBt.describe())

# Gráfico de comparación
plt.figure(figsize=(10, 5))
plt.boxplot([peso_Bt, peso_noBt], labels=['Bt', 'no Bt'])
plt.ylabel('Peso de mazorca (g)')
plt.title('Comparación de rendimiento entre híbridos')
plt.show()

# Prueba de normalidad
print("\nPrueba de normalidad (Shapiro-Wilk):")
stat_Bt, p_Bt = stats.shapiro(peso_Bt)
stat_noBt, p_noBt = stats.shapiro(peso_noBt)
print(f"Bt: p = {p_Bt:.4f}")
print(f"no Bt: p = {p_noBt:.4f}")

# Prueba de homocedasticidad
print("\nPrueba de homocedasticidad (Levene):")
stat, p = stats.levene(peso_Bt, peso_noBt)
print(f"p = {p:.4f}")

# Prueba t independiente
print("\nPrueba t para muestras independientes:")
tstat, pval_rend, df = ttest_ind(peso_Bt, peso_noBt, alternative='larger')
print(f"Estadístico t = {tstat:.4f}")
print(f"Valor p = {pval_rend:.5f}")
print(f"Grados de libertad = {df}")

# Tamaño del efecto (Cohen's d)
pooled_std = np.sqrt(((len(peso_Bt)-1)*np.std(peso_Bt, ddof=1)**2 + ((len(peso_noBt)-1)*np.std(peso_noBt, ddof=1)**2) / (len(peso_Bt) + len(peso_noBt) - 2))
d = (np.mean(peso_Bt) - np.mean(peso_noBt)) / pooled_std
print(f"\nTamaño del efecto (Cohen's d): {d:.3f}"))

# 3. Informe técnico

print("\n\n3. INFORME TÉCNICO")
print("------------------")
print("CONCLUSIONES DEL ESTUDIO:\n")

print("1. Resistencia al ataque de Diatraea saccharalis:")
if pval < 0.05:
    print(f"- Se encontró evidencia estadísticamente significativa (p = {pval:.5f}) de que")
    print(f"  el híbrido Bt presenta menor proporción de plantas dañadas ({p_Bt:.3f}) que")
    print(f"  el híbrido no Bt ({p_noBt:.3f}).")
    print(f"- El riesgo relativo de daño fue de {rr:.3f}, indicando que el maíz Bt tuvo")
    print(f"  un {((1-rr)*100):.1f}% menos de probabilidad de ser dañado.")
else:
    print("- No se encontró evidencia estadísticamente significativa de diferencia en la")
    print("  resistencia al ataque entre los híbridos.")

print("\n2. Rendimiento (peso de mazorca):")
if pval_rend < 0.05:
    print(f"- El híbrido Bt mostró un rendimiento significativamente mayor (p = {pval_rend:.5f})")
    print(f"  con un peso promedio de {np.mean(peso_Bt):.3f}g vs {np.mean(peso_noBt):.3f}g del no Bt.")
    print(f"- El tamaño del efecto fue {d:.3f}, considerado {'pequeño' if abs(d)<0.5 else 'mediano' if abs(d)<0.8 else 'grande'}.")
else:
    print("- No se encontraron diferencias significativas en el rendimiento entre los híbridos.")

print("\nLimitaciones:")
print("- Tamaño muestral relativamente pequeño para el análisis de rendimiento (9 parcelas por grupo)")
print("- No se controlaron completamente otras variables que podrían afectar el rendimiento")
print("- El estudio se realizó en una sola localidad y temporada")

print("\nRecomendaciones:")
print("- Continuar el monitoreo de posibles resistencias en poblaciones de Diatraea")
print("- Realizar estudios a mayor escala y en diferentes regiones")
print("- Evaluar el impacto económico de la adopción de maíz Bt considerando costos y beneficios")