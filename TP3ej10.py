import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# 1. Configuración de los datos
print("1. Configuración del estudio:")
print("- Diseño: Ensayo aleatorizado balanceado")
print("- Población: 400 pacientes en terapia intensiva (≥18 años, ventilación mecánica <96hs)")
print("- Grupos comparados: Midazolam vs Dexmedetomidina")
print("- Variable resultado: Presencia de agitación/delirio a las 24h")

# 2. Datos del estudio
n_total = 400
n_midazolam = n_total // 2
n_dexmed = n_total // 2
casos_midazolam = 136
casos_dexmed = 91

print(f"\nDistribución de pacientes:")
print(f"- Midazolam: {n_midazolam} pacientes, {casos_midazolam} con agitación/delirio")
print(f"- Dexmedetomidina: {n_dexmed} pacientes, {casos_dexmed} con agitación/delirio")

# 3. Cálculo de proporciones
p_midazolam = casos_midazolam / n_midazolam
p_dexmed = casos_dexmed / n_dexmed
diferencia = p_dexmed - p_midazolam  # Dexmed - Midazolam

print(f"\nTasas de complicaciones:")
print(f"- Midazolam: {p_midazolam:.3f} ({casos_midazolam}/{n_midazolam})")
print(f"- Dexmedetomidina: {p_dexmed:.3f} ({casos_dexmed}/{n_dexmed})")
print(f"- Diferencia absoluta: {diferencia:.3f}")

# 4. Prueba de hipótesis
print("\n2. Prueba de hipótesis:")
# H0: p_midazolam = p_dexmed (no hay diferencia en tasas de complicaciones)
# H1: p_midazolam ≠ p_dexmed (hay diferencia)

count = np.array([casos_midazolam, casos_dexmed])
nobs = np.array([n_midazolam, n_dexmed])
zstat, pval = proportions_ztest(count, nobs, alternative='two-sided')

print(f"Estadístico z = {zstat:.4f}")
print(f"Valor p = {pval:.6f}")

# 5. Intervalos de confianza
print("\n3. Intervalos de confianza 95%:")
# Para cada grupo
ci_midazolam = proportion_confint(casos_midazolam, n_midazolam, alpha=0.05, method='normal')
ci_dexmed = proportion_confint(casos_dexmed, n_dexmed, alpha=0.05, method='normal')

print(f"Midazolam: [{ci_midazolam[0]:.3f}, {ci_midazolam[1]:.3f}]")
print(f"Dexmedetomidina: [{ci_dexmed[0]:.3f}, {ci_dexmed[1]:.3f}]")

# Para la diferencia
se = np.sqrt(p_midazolam*(1-p_midazolam)/n_midazolam + p_dexmed*(1-p_dexmed)/n_dexmed)
ci_diff_low = diferencia - 1.96*se
ci_diff_upp = diferencia + 1.96*se

print(f"\nDiferencia (Dexmed - Midazolam): {diferencia:.3f}")
print(f"IC 95% para la diferencia: [{ci_diff_low:.3f}, {ci_diff_upp:.3f}]")

# 6. Riesgo relativo y NNT
rr = p_dexmed / p_midazolam
nnt = 1 / (p_midazolam - p_dexmed)

print(f"\nMedidas de impacto:")
print(f"- Riesgo relativo (RR): {rr:.3f}")
print(f"- Número necesario a tratar (NNT): {abs(round(nnt))} pacientes para prevenir un caso")

# 7. Visualización de resultados
plt.figure(figsize=(12, 5))

# Gráfico de barras con tasas
plt.subplot(1, 2, 1)
groups = ['Midazolam', 'Dexmedetomidina']
rates = [p_midazolam, p_dexmed]
cis = [[p_midazolam - ci_midazolam[0], ci_midazolam[1] - p_midazolam],
       [p_dexmed - ci_dexmed[0], ci_dexmed[1] - p_dexmed]]

plt.bar(groups, rates, yerr=np.array(cis).T, capsize=10, color=['skyblue', 'lightgreen'])
plt.ylabel('Proporción de pacientes con agitación/delirio')
plt.title('Comparación de tasas de complicaciones\ncon intervalos de confianza 95%')
plt.ylim(0, 0.8)

# Añadir valores exactos
for i, rate in enumerate(rates):
    plt.text(i, rate + 0.03, f"{rate:.3f}", ha='center')

# Gráfico de diferencia
plt.subplot(1, 2, 2)
plt.errorbar(0, diferencia, yerr=[[diferencia - ci_diff_low], [ci_diff_upp - diferencia]], 
             fmt='o', capsize=5, color='red', markersize=10)
plt.axhline(0, color='gray', linestyle='--')
plt.xlim(-0.5, 0.5)
plt.xticks([])
plt.ylabel('Diferencia en tasas\n(Dexmed - Midazolam)')
plt.title('Efecto de la dexmedetomidina\nvs midazolam con IC 95%')
plt.ylim(-0.3, 0.1)

# Añadir valor exacto
plt.text(0, diferencia - 0.05, f"{diferencia:.3f}\n(IC95%: [{ci_diff_low:.3f}, {ci_diff_upp:.3f}])", 
         ha='center', va='top')

plt.tight_layout()
plt.show()

# 8. Conclusión
print("\n4. Conclusiones:")
if pval < 0.05:
    print("- Existe una diferencia estadísticamente significativa (p = {:.6f}) en las tasas de".format(pval))
    print("  agitación/delirio entre los dos grupos de sedación.")
else:
    print("- No se encontró evidencia estadísticamente significativa (p = {:.6f}) de diferencia".format(pval))
    print("  en las tasas de agitación/delirio entre los grupos.")

print("\nHallazgos clave:")
print(f"- Reducción absoluta del riesgo: {abs(diferencia):.3f} (IC95%: [{abs(ci_diff_low):.3f}, {abs(ci_diff_upp):.3f}])")
print(f"- Reducción relativa del riesgo: {(1-rr)*100:.1f}%")
print(f"- NNT: {abs(round(nnt))} (pacientes a tratar con dexmedetomidina en lugar de midazolam")
print("  para prevenir un caso de agitación/delirio)")

print("\nLimitaciones:")
print("- Ensayo con seguimiento a 24h solamente")
print("- No se consideraron otros posibles efectos adversos")
print("- No se evaluó la eficacia sedante comparativa")