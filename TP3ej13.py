import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Configuración de los datos
print("1. Planteo del estudio:")
print("- Objetivo: Evaluar el efecto del entrenamiento de resistencia")
print("- Variable: Frecuencia cardíaca en reposo (lpm)")
print("- Diseño: Medidas repetidas (antes/después)")
print("- Muestra: 10 individuos")

# 2. Datos del estudio
data = {
    'Individuo': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Antes': [64, 84, 92, 64, 66, 72, 64, 66, 69, 72],
    'Después': [62, 72, 75, 66, 60, 65, 56, 57, 61, 59]
}

df = pd.DataFrame(data)

# 3. Cálculo de diferencias
df['Diferencia'] = df['Antes'] - df['Después']
print("\nDatos con diferencias calculadas:")
print(df)

# 4. Planteo de hipótesis
print("\n2. Planteo de hipótesis:")
print("- H0: El entrenamiento NO reduce la frecuencia cardíaca en reposo (μ_d ≤ 0)")
print("- H1: El entrenamiento SÍ reduce la frecuencia cardíaca en reposo (μ_d > 0)")
print("\nTipo de prueba:")
print("- Prueba t pareada para muestras relacionadas")

# 5. Análisis descriptivo
print("\n3. Estadísticos descriptivos:")
print("Antes del entrenamiento:")
print(df['Antes'].describe())
print("\nDespués del entrenamiento:")
print(df['Después'].describe())
print("\nDiferencias (Antes - Después):")
print(df['Diferencia'].describe())

# 6. Visualización de datos
plt.figure(figsize=(12, 5))

# Gráfico de líneas para cada individuo
plt.subplot(1, 2, 1)
for i in range(len(df)):
    plt.plot(['Antes', 'Después'], [df.loc[i, 'Antes'], df.loc[i, 'Después']], 
             marker='o', label=f'Ind {i+1}')
plt.ylabel('Frecuencia cardíaca (lpm)')
plt.title('Cambios individuales')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Gráfico de cajas comparativo
plt.subplot(1, 2, 2)
plt.boxplot([df['Antes'], df['Después']], labels=['Antes', 'Después'])
plt.ylabel('Frecuencia cardíaca (lpm)')
plt.title('Comparación general')

plt.tight_layout()
plt.show()

# 7. Verificación de supuestos
print("\n4. Verificación de supuestos:")

# Normalidad de las diferencias
stat, p_normalidad = stats.shapiro(df['Diferencia'])
print(f"Prueba de normalidad (Shapiro-Wilk): p = {p_normalidad:.4f}")

# Gráfico Q-Q para normalidad
plt.figure(figsize=(6, 4))
stats.probplot(df['Diferencia'], plot=plt)
plt.title('Gráfico Q-Q de las diferencias')
plt.show()

# 8. Prueba estadística
print("\n5. Prueba t pareada:")
t_stat, p_valor = stats.ttest_rel(df['Antes'], df['Después'], alternative='greater')

print(f"Estadístico t = {t_stat:.4f}")
print(f"Valor p = {p_valor:.5f}")

# 9. Tamaño del efecto
mean_diff = np.mean(df['Diferencia'])
sd_diff = np.std(df['Diferencia'], ddof=1)
cohen_d = mean_diff / sd_diff

print(f"\nTamaño del efecto (Cohen's d): {cohen_d:.3f}")

# Interpretación del tamaño del efecto
if cohen_d < 0.2:
    interpretacion = "efecto muy pequeño"
elif cohen_d < 0.5:
    interpretacion = "efecto pequeño"
elif cohen_d < 0.8:
    interpretacion = "efecto mediano"
else:
    interpretacion = "efecto grande"

print(f"Interpretación: {interpretacion}")

# 10. Intervalo de confianza para la diferencia media
ci = stats.t.interval(0.95, len(df['Diferencia'])-1, 
                      loc=mean_diff, scale=stats.sem(df['Diferencia']))

print(f"\nDiferencia media: {mean_diff:.2f} lpm")
print(f"IC 95%: [{ci[0]:.2f}, {ci[1]:.2f}]")

# 11. Conclusión
print("\n6. Conclusión:")
if p_valor < 0.05:
    print(f"- Existe evidencia estadísticamente significativa (p = {p_valor:.5f}) de que")
    print(f"  el entrenamiento reduce la frecuencia cardíaca en reposo.")
    print(f"- La reducción promedio fue de {mean_diff:.1f} lpm (IC95%: {ci[0]:.1f} a {ci[1]:.1f}).")
    print(f"- El tamaño del efecto fue {interpretacion} (d = {cohen_d:.2f}).")
else:
    print("- No se encontró evidencia suficiente (p = {p_valor:.5f}) para afirmar que")
    print("  el entrenamiento reduce la frecuencia cardíaca en reposo.")

print("\nLimitaciones:")
print("- Tamaño muestral pequeño (n=10)")
print("- No se controló el efecto de otras variables (edad, condición inicial, etc.)")
print("- No se evaluó la duración óptima del entrenamiento")