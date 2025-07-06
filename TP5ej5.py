import math

# Parámetros dados
diferencia_esperada = 0.5  # diferencia entre tratamientos en log10 copias/ml
desvio_estandar = 1        # desvío estándar común
alfa = 0.05                # nivel de significación
potencia = 0.80            # potencia deseada
perdidas_esperadas = 0.20  # 20% de pérdida esperada

# Z-scores
from scipy.stats import norm
z_alfa = norm.ppf(1 - alfa / 2)   # dos colas
z_beta = norm.ppf(potencia)

# Cálculo del tamaño muestral por grupo (comparación de dos medias)
numerador = (z_alfa + z_beta) ** 2 * 2 * desvio_estandar**2
n_sin_ajuste = numerador / diferencia_esperada**2
n_ajustado = n_sin_ajuste / (1 - perdidas_esperadas)

print(f"Tamaño de muestra por grupo sin ajuste por pérdidas: {math.ceil(n_sin_ajuste)}")
print(f"Tamaño de muestra por grupo ajustado por 20% de pérdidas: {math.ceil(n_ajustado)}")

# --------------------------------
# Materiales y Métodos
# --------------------------------
print("""
Materiales y Métodos:

Se determinó el tamaño muestral necesario para detectar una diferencia de 0,5 log10 copias/ml de carga viral
entre tratamientos con un desvío estándar común de 1, una potencia del 80% y un nivel de significación del 5%.
El tamaño calculado por grupo fue ajustado considerando una pérdida del 20% de los pacientes a lo largo del ensayo.
""")
