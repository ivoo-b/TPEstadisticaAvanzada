# problema4_tp2.py
from scipy.stats import norm
import numpy as np

# datos del estudio
n = 755
infectados = 203
p_hat = infectados / n
nivel_confianza = 0.95

# IC para proporción
z = norm.ppf(1 - (1 - nivel_confianza) / 2)
error_estandar = np.sqrt(p_hat * (1 - p_hat) / n)
LI = p_hat - z * error_estandar
LS = p_hat + z * error_estandar

print(f"Proporción muestral: {p_hat:.4f}")
print(f"IC del 95%: ({LI:.4f}, {LS:.4f})")

# nueva n para reducir la amplitud a la mitad
amplitud_actual = LS - LI
amplitud_deseada = amplitud_actual / 2
n_nuevo = (z ** 2 * p_hat * (1 - p_hat)) / (amplitud_deseada / 2) ** 2
print(f"Tamaño muestral necesario para reducir la amplitud a la mitad: {int(np.ceil(n_nuevo))}")

# inferencia del nivel de confianza del investigador
# IC: 0.2423 - 0.2954 → media = 0.26885, amplitud = 0.0531
media_ic = (0.2423 + 0.2954) / 2
amplitud = 0.2954 - media_ic
z_inferido = amplitud / np.sqrt(p_hat * (1 - p_hat) / n)
conf_inferido = norm.cdf(z_inferido) * 2 - 1

print(f"\nEl investigador trabajó con un nivel de confianza ≈ {conf_inferido*100:.2f}%")
