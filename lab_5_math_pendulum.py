import numpy as np
import matplotlib.pyplot as plt

# Для этой модели используем метод Рунге Кутта, так как метод Эйлера
# плохо подходит для системы с нелинейной динамикой

g = 9.81        # Ускорение свободного падения (м/с^2)
L = 1.0         # Длина маятника (м)
theta0 = np.pi / 4  # Начальный угол (рад)
omega0 = 0.0    # Начальная угловая скорость (рад/с)
t_max = 10      # Время моделирования (с)
dt = 0.01       # Шаг по времени (с)

t = np.arange(0, t_max, dt)

# Инициализация массивов для угла и угловой скорости
theta = np.zeros_like(t)
omega = np.zeros_like(t)

# Начальные условия
theta[0] = theta0
omega[0] = omega0

# Численное решение (метод Рунге-Кутты 4-го порядка)
for i in range(1, len(t)):
    def derivatives(theta, omega):
        return omega, - (g / L) * np.sin(theta)

    k1_theta, k1_omega = derivatives(theta[i-1], omega[i-1])
    k2_theta, k2_omega = derivatives(theta[i-1] + 0.5 * dt * k1_theta, omega[i-1] + 0.5 * dt * k1_omega)
    k3_theta, k3_omega = derivatives(theta[i-1] + 0.5 * dt * k2_theta, omega[i-1] + 0.5 * dt * k2_omega)
    k4_theta, k4_omega = derivatives(theta[i-1] + dt * k3_theta, omega[i-1] + dt * k3_omega)

    theta[i] = theta[i-1] + (dt / 6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
    omega[i] = omega[i-1] + (dt / 6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)

# Аналитическое решение
omega_analytical = np.sqrt(g / L)
theta_analytical = theta0 * np.cos(omega_analytical * t) + (omega0 / omega_analytical) * np.sin(omega_analytical * t)

plt.figure(figsize=(10, 6))
plt.plot(t, theta, label='Численное решение (Угол)', color='blue')
plt.plot(t, theta_analytical, label='Аналитическое решение (Угол)', color='red', linestyle='--')
plt.title('Динамика математического маятника')
plt.xlabel('Время (с)')
plt.ylabel('Угол (рад)')
plt.axhline(0, color='gray', linestyle=':')
plt.legend()
plt.grid()
plt.show()
