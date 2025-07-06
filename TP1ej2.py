# Solucionar error anterior: redefinir 'os' y repetir proceso

import os
import pandas as pd
import numpy as np
from scipy import stats

# Rutas de los archivos
ruta_datos = r"DatosCSV"
ruta_tp2 = r"TP2"

# Crear carpeta TP2 si no existe
os.makedirs(ruta_tp2, exist_ok=True)

# Problema 2 - Intervalos de confianza con girasol.csv
archivo = os.path.join(ruta_datos, "girasol.csv")
datos = pd.read_csv(archivo)

# Asegurar que la columna tenga nombre correcto
columna = datos.columns[0]
variable = datos[columna]

# 1. Estimar media con IC del 95%
media = np.mean(variable)
desvio = np.std(variable, ddof=1)
n = len(variable)
error_estandar = desvio / np.sqrt(n)
z_critico_95 = stats.norm.ppf(0.975)
ic_95 = (media - z_critico_95 * error_estandar, media + z_critico_95 * error_estandar)

# 2. Estimar media con IC del 99%
z_critico_99 = stats.norm.ppf(0.995)
ic_99 = (media - z_critico_99 * error_estandar, media + z_critico_99 * error_estandar)

# 3. Tamaño muestral necesario para reducir amplitud a la mitad (IC 99%)
amplitud_actual = ic_99[1] - ic_99[0]
amplitud_deseada = amplitud_actual / 2
n_nuevo = (z_critico_99 * desvio / (amplitud_deseada / 2)) ** 2
n_nuevo = int(np.ceil(n_nuevo))

# 4. Estimar proporción de alícuotas de calidad inferior (fosfolípidos > 0.40)
proporcion_mala_calidad = np.mean(variable > 0.40)
error_estandar_p = np.sqrt(proporcion_mala_calidad * (1 - proporcion_mala_calidad) / n)
z_critico_p = stats.norm.ppf(0.975)
ic_proporcion = (
    proporcion_mala_calidad - z_critico_p * error_estandar_p,
    proporcion_mala_calidad + z_critico_p * error_estandar_p,
)

# Guardar el código en un archivo .py
codigo = f"""
import pandas as pd
import numpy as np
from scipy import stats

datos = pd.read_csv(r"{archivo}")
variable = datos.iloc[:, 0]

media = np.mean(variable)
desvio = np.std(variable, ddof=1)
n = len(variable)
error_estandar = desvio / np.sqrt(n)

z_95 = stats.norm.ppf(0.975)
ic_95 = (media - z_95 * error_estandar, media + z_95 * error_estandar)

z_99 = stats.norm.ppf(0.995)
ic_99 = (media - z_99 * error_estandar, media + z_99 * error_estandar)

amplitud_actual = ic_99[1] - ic_99[0]
amplitud_deseada = amplitud_actual / 2
n_nuevo = int(np.ceil((z_99 * desvio / (amplitud_deseada / 2)) ** 2))

proporcion_mala_calidad = np.mean(variable > 0.40)
error_estandar_p = np.sqrt(proporcion_mala_calidad * (1 - proporcion_mala_calidad) / n)
z_p = stats.norm.ppf(0.975)
ic_proporcion = (
    proporcion_mala_calidad - z_p * error_estandar_p,
    proporcion_mala_calidad + z_p * error_estandar_p,
)

print("Media:", round(media, 4))
print("IC 95%:", tuple(round(x, 4) for x in ic_95))
print("IC 99%:", tuple(round(x, 4) for x in ic_99))
print("Nuevo tamaño muestral necesario:", n_nuevo)
print("Proporción de mala calidad:", round(proporcion_mala_calidad, 4))
print("IC 95% para proporción:", tuple(round(x, 4) for x in ic_proporcion))
"""

# Escribir el archivo
ruta_archivo = os.path.join(ruta_tp2, "problema2.py")
with open(ruta_archivo, "w", encoding="utf-8") as f:
    f.write(codigo)

import ace_tools as tools; tools.display_dataframe_to_user(name="Datos girasol.csv", dataframe=datos)