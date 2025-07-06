import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Cargar los datos
datos = pd.read_csv('preclampsia.csv')

# 1. Análisis descriptivo
print("Resumen estadístico por grupo:")
print(datos.groupby('Grupo').describe())

# 2. Visualización de los datos
plt.figure(figsize=(12, 5))

# Gráfico para leptina
plt.subplot(1, 2, 1)
sns.boxplot(x='Grupo', y='leptina', data=datos)
plt.title('Expresión de leptina')
plt.ylabel('Unidades arbitrarias (log2)')

# Gráfico para fibulina
plt.subplot(1, 2, 2)
sns.boxplot(x='Grupo', y='fibulina', data=datos)
plt.title('Expresión de fibulina')
plt.ylabel('Unidades arbitrarias (log2)')

plt.tight_layout()
plt.show()

# 3. Prueba de normalidad (Shapiro-Wilk)
print("\nPrueba de normalidad (Shapiro-Wilk):")
for proteina in ['leptina', 'fibulina']:
    for grupo in datos['Grupo'].unique():
        stat, p = stats.shapiro(datos[datos['Grupo'] == grupo][proteina])
        print(f"{proteina} - {grupo}: p = {p:.4f}")

# 4. Prueba de homocedasticidad (Levene)
print("\nPrueba de homocedasticidad (Levene):")
for proteina in ['leptina', 'fibulina']:
    stat, p = stats.levene(datos[datos['Grupo'] == 'Control'][proteina],
                          datos[datos['Grupo'] == 'PE'][proteina])
    print(f"{proteina}: p = {p:.4f}")

# 5. Comparación entre grupos (t-test o Mann-Whitney)
print("\nComparación entre grupos:")
resultados = {}

for proteina in ['leptina', 'fibulina']:
    # Asumiendo que algunos datos no son normales, usamos Mann-Whitney
    stat, p = stats.mannwhitneyu(datos[datos['Grupo'] == 'Control'][proteina],
                                datos[datos['Grupo'] == 'PE'][proteina],
                                alternative='two-sided')
    
    resultados[proteina] = {
        'Estadístico': stat,
        'p-valor': p,
        'Control_media': datos[datos['Grupo'] == 'Control'][proteina].mean(),
        'PE_media': datos[datos['Grupo'] == 'PE'][proteina].mean(),
        'Control_mediana': datos[datos['Grupo'] == 'Control'][proteina].median(),
        'PE_mediana': datos[datos['Grupo'] == 'PE'][proteina].median()
    }
    
    print(f"\n{proteina}:")
    print(f"Mann-Whitney U = {stat}, p = {p:.4f}")
    print(f"Mediana Control: {resultados[proteina]['Control_mediana']:.2f}")
    print(f"Mediana PE: {resultados[proteina]['PE_mediana']:.2f}")

# 6. Visualización de resultados significativos
significativos = {k:v for k,v in resultados.items() if v['p-valor'] < 0.05}
if significativos:
    print("\nDiferencias significativas encontradas en:")
    for proteina, res in significativos.items():
        print(f"- {proteina} (p = {res['p-valor']:.4f})")
        
        # Gráfico de barras con error estándar
        plt.figure(figsize=(6, 4))
        sns.barplot(x='Grupo', y=proteina, data=datos, ci=68, capsize=0.1)
        plt.title(f'Expresión de {proteina}*')
        plt.ylabel('Unidades arbitrarias (log2)')
        plt.text(0.5, 0.9, f'p = {res["p-valor"]:.4f}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.show()
else:
    print("\nNo se encontraron diferencias significativas")