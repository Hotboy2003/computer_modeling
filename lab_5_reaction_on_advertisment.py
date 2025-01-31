import numpy as np
import matplotlib.pyplot as plt

r = 0.1       # Скорость роста
K = 100       # Поддерживающая ёмкость
P0 = 10       # Начальная численность
t = np.linspace(0, 100, 200)

# Аналитическое решение
P_analytical = (K * P0 * np.exp(r * t)) / (K + P0 * (np.exp(r * t) - 1))

# Численное решение
dt = t[1] - t[0]
P_numerical = np.zeros_like(t)
P_numerical[0] = P0

for i in range(1, len(t)):
    P_numerical[i] = P_numerical[i-1] + r * P_numerical[i-1] * (1 - P_numerical[i-1] / K) * dt

plt.figure(figsize=(10, 6))
plt.plot(t, P_analytical, label='Аналитическое решение', color='blue')
plt.plot(t, P_numerical, label='Численное решение', color='red', linestyle='--')
plt.title('Моделирование логистического роста популяции')
plt.xlabel('Время (дни)')
plt.ylabel('Численность популяции')
plt.axhline(y=K, color='green', linestyle=':', label='Поддерживающая ёмкость (K)')
plt.legend()
plt.grid()
plt.show()