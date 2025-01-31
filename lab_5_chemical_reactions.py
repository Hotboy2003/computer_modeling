import numpy as np
import matplotlib.pyplot as plt

a = 5.0  # Концентрация A (моль/л)
b = 8.0  # Концентрация B (моль/л)
k = 0.1  # Константа скорости (1/с)
t_max = 5  # Максимальное время (с)
dt = 0.1  # Шаг по времени (с)

t = np.arange(0, t_max, dt)

# Аналитическое решение
x_t = a * b * (1 - np.exp(-k * (b - a) * t)) / (b - a * np.exp(-k * (b - a) * t))

# Численное решение методом Эйлера
x_euler = np.zeros_like(t)
x_euler[0] = 0

for i in range(1, len(t)):
    dx_dt = k * (a - x_euler[i-1]) * (b - x_euler[i-1])  # Скорость реакции
    x_euler[i] = x_euler[i-1] + dx_dt * dt

plt.figure(figsize=(10, 6))
plt.plot(t, x_t, label='Аналитическое решение x(t)', color='blue')
plt.plot(t, x_euler, label='Численное решение (Эйлер)', color='orange', linestyle='--')
plt.axhline(a, color='red', linestyle='--', label='Концентрация A (a)')
plt.axhline(b, color='green', linestyle='--', label='Концентрация B (b)')
plt.title('Аналитическое и численное решение уравнения x(t)')
plt.xlabel('Время (с)')
plt.ylabel('Концентрация (моль/л)')
plt.legend()
plt.grid()
plt.xlim(0, t_max)
plt.ylim(0, max(a, b) + 5)
plt.show()
