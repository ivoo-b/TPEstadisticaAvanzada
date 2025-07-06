 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

# 1. Identificación de variables y muestras
print("1. Identificación de variables:")
print("- Unidad experimental: cada oveja Merino")
print("- Muestra 1: 70 hembras sin tratamiento hormonal")
print("- Muestra 2: 40 hembras con sincronización hormonal")
print("- Población: todas las ovejas Merino en condiciones similares")
print("- Variable aleatoria: resultado binario (preñada/no preñada)")

# 2. Datos del estudio
n1 = 70  # Grupo sin tratamiento
x1 = 56  # Preñadas en grupo 1
n2 = 40  # Grupo con tratamiento
x2 = 35  # Preñadas en grupo 2

# Tasas observadas
p1 = x1/n1
p2 = x2/n2

print(f"\nTasas observadas:")
print(f"- Grupo control (sin tratamiento): {p1:.3f} ({x1}/{n1})")
print(f"- Grupo tratamiento (con sincronización): {p2:.3f} ({x2}/{n2})")

# 3. Prueba de hipótesis para dos proporciones
print("\n2. Prueba de hipótesis:")
# H0: p1 = p2 (no hay diferencia en tasas de preñez)
# H1: p1 < p2 (el tratamiento aumenta la tasa de preñez)

# Realizamos prueba z para dos proporciones
count = np.array([x1, x2])
nobs = np.array([n1, n2])
zstat, pval = proportions_ztest(count, nobs, alternative='smaller')

print(f"Estadístico z = {zstat:.4f}")
print(f"Valor p = {pval:.5f}")

# 4. Intervalo de confianza para el grupo con tratamiento
print("\n3. Intervalo de confianza 95% para grupo con tratamiento:")
ci_low, ci_upp = proportion_confint(x2, n2, alpha=0.05, method='normal')
print(f"Tasa estimada: {p2:.3f}")
print(f"IC 95%: [{ci_low:.3f}, {ci_upp:.3f}]")

# 5. Diferencia de proporciones y su IC
print("\nDiferencia en tasas (tratamiento - control):")
diff = p2 - p1
se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
ci_diff_low = diff - 1.96*se
ci_diff_upp = diff + 1.96*se

print(f"Diferencia: {diff:.3f}")
print(f"IC 95% para la diferencia: [{ci_diff_low:.3f}, {ci_diff_upp:.3f}]")

# 6. Visualización de resultados
plt.figure(figsize=(10, 5))

# Gráfico de barras con tasas
plt.subplot(1, 2, 1)
groups = ['Sin tratamiento', 'Con tratamiento']
rates = [p1, p2]
cis = [(p1 - proportion_confint(x1, n1, alpha=0.05, method='normal'))[0], 
       proportion_confint(x1, n1, alpha=0.05, method='normal')[1] - p1],
      [(p2 - proportion_confint(x2, n2, alpha=0.05, method='normal'))[0], 
       proportion_confint(x2, n2, alpha=0.05, method='normal')[1] - p2]

plt.bar(groups, rates, yerr=np.array(cis).T, capsize=10, color=['skyblue', 'lightgreen'])
plt.ylabel('Tasa de preñez')
plt.title('Comparación de tasas de preñez\ncon intervalos de confianza 95%')
plt.ylim(0, 1.1)

# Añadir valores exactos
for i, rate in enumerate(rates):
    plt.text(i, rate + 0.05, f"{rate:.3f}", ha='center')

# Gráfico de diferencia
plt.subplot(1, 2, 2)
plt.errorbar(0, diff, yerr=[[diff - ci_diff_low], [ci_diff_upp - diff]], 
             fmt='o', capsize=5, color='red')
plt.axhline(0, color='gray', linestyle='--')
plt.xlim(-0.5, 0.5)
plt.xticks([])
plt.ylabel('Diferencia en tasas\n(Tratamiento - Control)')
plt.title('Efecto del tratamiento hormonal\ncon IC 95%')
plt.ylim(-0.2, 0.4)

# Añadir valor exacto
plt.text(0, diff + 0.02, f"{diff:.3f}", ha='center')

plt.tight_layout()
plt.show()

# 7. Conclusión
print("\n4. Conclusiones:")
if pval < 0.05:
    print(f"- Existe evidencia estadística (p = {pval:.5f} < 0.05) que respalda que la sincronización hormonal")
    print("  aumenta significativamente la tasa de preñez en ovejas Merino.")
else:
    print(f"- No hay evidencia suficiente (p = {pval:.5f} > 0.05) para afirmar que la sincronización hormonal")
    print("  afecta la tasa de preñez en ovejas Merino.")

print(f"- La tasa de preñez con tratamiento fue de {p2:.3f} (IC 95%: [{ci_low:.3f}, {ci_upp:.3f}])")
print(f"- La diferencia estimada fue de {diff:.3f} (IC 95%: [{ci_diff_low:.3f}, {ci_diff_upp:.3f}])")
print("- Error posible: Error Tipo I (falso positivo) si rechazamos H0 siendo verdadera, con probabilidad α=0.05")