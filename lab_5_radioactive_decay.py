import numpy as np
import matplotlib.pyplot as plt

lambda_decay = 0.1
N0 = 1000
t = np.linspace(0, 50, 100)

# Аналитическое решение
N_analytical = N0 * np.exp(-lambda_decay * t)

# Численное решение
dt = t[1] - t[0]
N_numerical = np.zeros_like(t)
N_numerical[0] = N0

for i in range(1, len(t)):
    N_numerical[i] = N_numerical[i-1] - lambda_decay * N_numerical[i-1] * dt

plt.figure(figsize=(10, 6))
plt.plot(t, N_analytical, label='Аналитическое решение', color='blue')
plt.plot(t, N_numerical, label='Численное решение', color='red', linestyle='--')
plt.title('Сравнение аналитического и численного решений для радиоактивного распада')
plt.xlabel('Время')
plt.ylabel('Количество атомов')
plt.legend()
plt.grid()
plt.show()