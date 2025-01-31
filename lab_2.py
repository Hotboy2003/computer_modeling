import pandas as pd
import numpy as np

def linear_interpolation(x, y):
    if len(x) != len(y):
        raise ValueError("Длины x и y должны совпадать.")

    interpolated = np.copy(y)

    nan_indices = np.where(np.isnan(y))[0]

    for idx in nan_indices:
        left_index = idx - 1
        right_index = idx + 1

        # Ищем ближайшие ненулевые значения
        while left_index >= 0 and np.isnan(y[left_index]):
            left_index -= 1
        while right_index < len(y) and np.isnan(y[right_index]):
            right_index += 1

        if left_index >= 0 and right_index < len(y):
            x_left = x[left_index]
            x_right = x[right_index]
            y_left = y[left_index]
            y_right = y[right_index]
            interpolated[idx] = y_left + (y_right - y_left) * ((x[idx] - x_left) / (x_right - x_left))

    return np.round(interpolated,1)

def quadratic_interpolation(x, y):
    if len(x) != len(y):
        raise ValueError("Длины x и y должны совпадать.")

    interpolated = np.copy(y)
    nan_indices = np.where(np.isnan(y))[0]

    for idx in nan_indices:
        left_index = idx - 1
        right_index = idx + 1

        while left_index >= 0 and np.isnan(y[left_index]):
            left_index -= 1
        while right_index < len(y) and np.isnan(y[right_index]):
            right_index += 1

        if left_index >= 0 and right_index < len(y):
            # Используем три точки для квадратичной интерполяции
            x0, x1, x2 = x[left_index - 1:left_index + 2] if left_index > 0 else x[left_index:right_index + 1]
            y0, y1, y2 = y[left_index - 1:left_index + 2] if left_index > 0 else y[left_index:right_index + 1]

            # Решаем систему уравнений для нахождения коэффициентов
            A = np.array([[1, x0, x0 ** 2],
                          [1, x1, x1 ** 2],
                          [1, x2, x2 ** 2]])
            B = np.array([y0, y1, y2])
            coeffs = np.linalg.solve(A, B)

            interpolated[idx] = coeffs[0] + coeffs[1] * x[idx] + coeffs[2] * x[idx] ** 2

    return np.round(interpolated,1)

def interpolate_and_save(file_path, interpolation_func, output_file_path):
    data = pd.read_csv(file_path, delimiter=',', decimal=',')
    days = data.iloc[:, 0].values
    result_data = pd.DataFrame({'Days': days})

    for col in data.columns[1:]:
        temperatures = data[col].values.astype(float)
        interpolated = interpolation_func(days, temperatures)
        result_data[col] = interpolated

    result_data.to_csv(output_file_path, index=False)
    print(f"Интерполяция завершена. Результаты сохранены в {output_file_path}")

file_path = 'Data_2.csv'
interpolate_and_save(file_path, linear_interpolation, 'interpolated_temperature_data_linear.csv')
interpolate_and_save(file_path, quadratic_interpolation, 'interpolated_temperature_data_quadratic.csv')