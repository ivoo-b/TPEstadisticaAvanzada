import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Crear el DataFrame con los datos
data = {
    'Dieta': ['Dieta 1']*8 + ['Dieta 2']*8,
    'Colesterol_aorta': [10.4, 14.2, 20.5, 19.6, 18.5, 24.0, 23.4, 13.6,
                        7.5, 7.2, 6.7, 7.6, 11.2, 9.6, 6.8, 8.5]
}

df = pd.DataFrame(data)

# 1. Análisis descriptivo
print("Resumen estadístico por grupo:")
print(df.groupby('Dieta').describe())

# 2. Visualización de los datos
plt.figure(figsize=(10, 5))

# Gráfico de cajas
plt.subplot(1, 2, 1)
sns.boxplot(x='Dieta', y='Colesterol_aorta', data=df)
plt.title('Contenido de colesterol en aorta')
plt.ylabel('mg/g')

# Gráfico de puntos
plt.subplot(1, 2, 2)
sns.stripplot(x='Dieta', y='Colesterol_aorta', data=df, jitter=True, palette='Set2')
plt.title('Distribución por individuo')
plt.ylabel('mg/g')

plt.tight_layout()
plt.show()

# 3. Prueba de normalidad (Shapiro-Wilk)
print("\nPrueba de normalidad (Shapiro-Wilk):")
for dieta in df['Dieta'].unique():
    stat, p = stats.shapiro(df[df['Dieta'] == dieta]['Colesterol_aorta'])
    print(f"{dieta}: p = {p:.4f}")

# 4. Prueba de homocedasticidad (Levene)
print("\nPrueba de homocedasticidad (Levene):")
stat, p = stats.levene(df[df['Dieta'] == 'Dieta 1']['Colesterol_aorta'],
                      df[df['Dieta'] == 'Dieta 2']['Colesterol_aorta'])
print(f"p = {p:.4f}")

# 5. Comparación entre grupos (t-test independiente o Mann-Whitney)
# Como los datos no son normales en ambos grupos (p < 0.05 en Shapiro-Wilk), usamos Mann-Whitney
print("\nPrueba de Mann-Whitney U:")
stat, p = stats.mannwhitneyu(df[df['Dieta'] == 'Dieta 1']['Colesterol_aorta'],
                            df[df['Dieta'] == 'Dieta 2']['Colesterol_aorta'],
                            alternative='two-sided')

print(f"Estadístico U = {stat}, p-valor = {p:.5f}")

# 6. Tamaño del efecto (Cliff's delta)
def cliffs_delta(x, y):
    """Calcula el tamaño del efecto Cliff's delta"""
    n = len(x)
    m = len(y)
    total = n * m
    count = 0
    
    for i in x:
        for j in y:
            if i > j:
                count += 1
            elif i < j:
                count -= 1
    
    delta = count / total
    return delta

diet1 = df[df['Dieta'] == 'Dieta 1']['Colesterol_aorta']
diet2 = df[df['Dieta'] == 'Dieta 2']['Colesterol_aorta']
delta = cliffs_delta(diet1, diet2)
print(f"\nTamaño del efecto (Cliff's delta): {delta:.3f}")

# Interpretación del tamaño del efecto
if abs(delta) < 0.147:
    interpretation = "efecto insignificante"
elif abs(delta) < 0.33:
    interpretation = "efecto pequeño"
elif abs(delta) < 0.474:
    interpretation = "efecto mediano"
else:
    interpretation = "efecto grande"

print(f"Interpretación: {interpretation}")

# 7. Gráfico de resultados con significancia
plt.figure(figsize=(8, 5))
sns.barplot(x='Dieta', y='Colesterol_aorta', data=df, ci=95, capsize=0.1)
plt.title('Contenido de colesterol en aorta\npor tipo de dieta')
plt.ylabel('mg/g')

# Añadir línea de significancia
y_max = df['Colesterol_aorta'].max() + 3
plt.plot([0, 1], [y_max, y_max], 'k', linewidth=1)
plt.text(0.5, y_max + 0.5, f'p = {p:.5f}', ha='center')

plt.show()
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