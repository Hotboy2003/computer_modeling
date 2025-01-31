import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.fft import fft
import matplotlib.pyplot as plt

file_path = 'Data_7.xlsx' 
data = pd.read_excel(file_path)

results = []
x = np.arange(len(data))

for column in data.columns[1:]:
    y = data[column].to_numpy() 

    valid_indices = np.where(y > 0.01)[0]
    y_filtered = y[valid_indices]
    x_filtered = x[valid_indices]

    # Линейная регрессия
    X = sm.add_constant(x_filtered)
    model = sm.OLS(y_filtered, X).fit()

    results.append({
        'Column': column,
        'Slope': model.params[1],
        'Intercept': model.params[0],
        'P-value': model.pvalues[1],
        'Significance': 'Significant' if model.pvalues[1] < 0.05 else 'Not Significant',
        'R-squared': model.rsquared
    })

    # Гармонический анализ
    N = len(y_filtered)
    T = 1.0 
    yf = fft(y_filtered)
    xf = np.fft.fftfreq(N, T / N)

    amplitudes = 2.0 / N * np.abs(yf[:N // 2])

    non_zero_indices = xf[:N // 2] > 0
    significant_harmonics = xf[:N // 2][non_zero_indices][amplitudes[non_zero_indices] > 0.1] 
    significant_amplitudes = amplitudes[non_zero_indices][amplitudes[non_zero_indices] > 0.1]

    # Построение модельного уравнения
    harmonic_terms = sum(
        [amplitude * np.sin(2 * np.pi * harmonic * x_filtered / N) for harmonic, amplitude in
         zip(significant_harmonics, significant_amplitudes)]
    )

    # Прогноз на год вперед
    future_steps = 365
    future_x = np.arange(len(data), len(data) + future_steps)
    future_X = sm.add_constant(future_x)
    future_trend = model.predict(future_X)
    future_harmonics = sum(
        [amplitude * np.sin(2 * np.pi * harmonic * future_x / N) for harmonic, amplitude in
         zip(significant_harmonics, significant_amplitudes)]
    )
    future_values = future_trend + future_harmonics

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(data['Date'], y, label='Исходные данные', color='blue')
    plt.plot(data['Date'].iloc[valid_indices], model.predict(X), label='Тренд (линейная регрессия)', color='orange',
             linestyle='--')
    plt.plot(pd.date_range(data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps), future_values,
             label='Прогноз', color='green', linestyle='--')
    plt.title(f'Визуализация тренда и прогноза для {column}')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.xlim(data['Date'].iloc[valid_indices].min(),
             data['Date'].iloc[valid_indices].max() + pd.Timedelta(days=future_steps))
    plt.legend()
    plt.grid()

    
    plt.subplot(1, 2, 2)
    plt.stem(significant_harmonics, significant_amplitudes)
    plt.title(f'Значимые гармоники для {column}')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.grid()

    plt.tight_layout()
    plt.show()

results_df = pd.DataFrame(results)
print(results_df)

significant_count = results_df['Significance'].value_counts().get('Significant', 0)
total_count = len(results_df)

print("\nОбщий вывод об оценке значимости:")
print(f"Количество значимых коэффициентов: {significant_count} из {total_count}")
