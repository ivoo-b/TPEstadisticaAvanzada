import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest

# 1. Configuración de los datos
print("1. Diseño del estudio:")
print("- Área de estudio: Alrededores del Lago Lolog, Neuquén")
print("- Especie invasora: Rosa mosqueta (especie exótica)")
print("- Tipos de parcelas: Con disturbio vs Sin disturbio")

# 2. Datos recolectados
n_con_disturbio = 23
n_sin_disturbio = 17
invadidas_con_disturbio = 15
invadidas_sin_disturbio = 10

print(f"\nDatos recolectados:")
print(f"- Parcelas con disturbio: {invadidas_con_disturbio}/{n_con_disturbio} invadidas")
print(f"- Parcelas sin disturbio: {invadidas_sin_disturbio}/{n_sin_disturbio} invadidas")

# 3. Cálculo de proporciones
p_con = invadidas_con_disturbio / n_con_disturbio
p_sin = invadidas_sin_disturbio / n_sin_disturbio
diferencia = p_con - p_sin

print(f"\nProporciones de invasión:")
print(f"- Con disturbio: {p_con:.3f} ({invadidas_con_disturbio}/{n_con_disturbio})")
print(f"- Sin disturbio: {p_sin:.3f} ({invadidas_sin_disturbio}/{n_sin_disturbio})")
print(f"- Diferencia: {diferencia:.3f}")

# 4. Prueba de hipótesis para dos proporciones
print("\n2. Prueba de hipótesis:")
# H0: p_con = p_sin (no hay asociación entre disturbio e invasión)
# H1: p_con ≠ p_sin (existe asociación)

count = np.array([invadidas_con_disturbio, invadidas_sin_disturbio])
nobs = np.array([n_con_disturbio, n_sin_disturbio])
zstat, pval = proportions_ztest(count, nobs, alternative='two-sided')

print(f"Estadístico z = {zstat:.4f}")
print(f"Valor p = {pval:.4f}")

# 5. Intervalos de confianza
print("\n3. Intervalos de confianza 95%:")
# Para cada grupo
ci_con = stats.proportion_confint(invadidas_con_disturbio, n_con_disturbio, alpha=0.05, method='normal')
ci_sin = stats.proportion_confint(invadidas_sin_disturbio, n_sin_disturbio, alpha=0.05, method='normal')

print(f"Con disturbio: [{ci_con[0]:.3f}, {ci_con[1]:.3f}]")
print(f"Sin disturbio: [{ci_sin[0]:.3f}, {ci_sin[1]:.3f}]")

# Para la diferencia
se = np.sqrt(p_con*(1-p_con)/n_con_disturbio + p_sin*(1-p_sin)/n_sin_disturbio)
ci_diff_low = diferencia - 1.96*se
ci_diff_upp = diferencia + 1.96*se

print(f"\nDiferencia (Con - Sin disturbio): {diferencia:.3f}")
print(f"IC 95% para la diferencia: [{ci_diff_low:.3f}, {ci_diff_upp:.3f}]")

# 6. Prueba Chi-cuadrado de independencia
print("\n4. Prueba Chi-cuadrado de independencia:")
tabla_contingencia = pd.DataFrame({
    'Invasión': [invadidas_con_disturbio, invadidas_sin_disturbio],
    'No invasión': [n_con_disturbio - invadidas_con_disturbio, n_sin_disturbio - invadidas_sin_disturbio]
}, index=['Con disturbio', 'Sin disturbio'])

chi2, p_chi, dof, expected = stats.chi2_contingency(tabla_contingencia)
print(f"Estadístico Chi2 = {chi2:.4f}")
print(f"Valor p = {p_chi:.4f}")

# 7. Visualización de resultados
plt.figure(figsize=(12, 5))

# Gráfico de barras con tasas
plt.subplot(1, 2, 1)
groups = ['Con disturbio', 'Sin disturbio']
rates = [p_con, p_sin]
cis = [[p_con - ci_con[0], ci_con[1] - p_con],
       [p_sin - ci_sin[0], ci_sin[1] - p_sin]]

bars = plt.bar(groups, rates, yerr=np.array(cis).T, capsize=10, color=['salmon', 'lightblue'])
plt.ylabel('Proporción de parcelas invadidas')
plt.title('Invasión por rosa mosqueta\nsegún presencia de disturbio')
plt.ylim(0, 1)

# Añadir valores exactos
for bar, rate in zip(bars, rates):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f"{rate:.2f}", ha='center')

# Gráfico de diferencia
plt.subplot(1, 2, 2)
plt.errorbar(0, diferencia, yerr=[[diferencia - ci_diff_low], [ci_diff_upp - diferencia]], 
             fmt='o', capsize=5, color='green', markersize=10)
plt.axhline(0, color='gray', linestyle='--')
plt.xlim(-0.5, 0.5)
plt.xticks([])
plt.ylabel('Diferencia en proporciones\n(Con - Sin disturbio)')
plt.title('Efecto del disturbio en la invasión\ncon IC 95%')
plt.ylim(-0.3, 0.5)

# Añadir valor exacto
plt.text(0, diferencia + 0.02, f"{diferencia:.2f}\n(IC95%: [{ci_diff_low:.2f}, {ci_diff_upp:.2f}])", 
         ha='center')

plt.tight_layout()
plt.show()

# 8. Conclusión
print("\n5. Conclusiones:")
if pval < 0.05:
    print("- Existe evidencia estadística (p = {:.4f}) que sugiere una asociación significativa".format(pval))
    print("  entre la presencia de disturbios y la invasión por rosa mosqueta.")
else:
    print("- No hay evidencia suficiente (p = {:.4f}) para afirmar que exista una asociación".format(pval))
    print("  entre disturbios y la invasión por rosa mosqueta.")

print("\nHallazgos clave:")
print(f"- Diferencia observada: {diferencia:.2f} (IC95%: [{ci_diff_low:.2f}, {ci_diff_upp:.2f}])")
print(f"- La prueba Chi-cuadrado confirma estos resultados (p = {p_chi:.4f})")

print("\nLimitaciones:")
print("- Tamaño muestral relativamente pequeño (40 parcelas en total)")
print("- No se discriminó entre tipos de disturbio (pastoreo, incendios, etc.)")
print("- Estudio observacional: no permite establecer causalidad")