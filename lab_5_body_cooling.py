import numpy as np
import matplotlib.pyplot as plt

k = 0.1
T_a = 20        # Температура окружающей среды (°C)
T0 = 100        # Начальная температура тела (°C)
t = np.linspace(0, 50, 100)

# Аналитическое решение
T_analytical = T_a + (T0 - T_a) * np.exp(-k * t)

# Численное решение
dt = t[1] - t[0]
T_numerical = np.zeros_like(t)
T_numerical[0] = T0

for i in range(1, len(t)):
    T_numerical[i] = T_numerical[i-1] - k * (T_numerical[i-1] - T_a) * dt

plt.figure(figsize=(10, 6))
plt.plot(t, T_analytical, label='Аналитическое решение', color='blue')
plt.plot(t, T_numerical, label='Численное решение', color='red', linestyle='--')
plt.title('Сравнение аналитического и численного решений для процесса охлаждения тела')
plt.xlabel('Время (с)')
plt.ylabel('Температура (°C)')
plt.legend()
plt.grid()
plt.show()