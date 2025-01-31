import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file_path = 'Data_4.xlsx'
data = pd.read_excel(file_path)
columns_of_interest = data.iloc[:, 1:4]

statistics = {
    'Среднее': columns_of_interest.mean(),
    'Стандартное отклонение': columns_of_interest.std(),
    'Медиана': columns_of_interest.median(),
    'Мода': columns_of_interest.mode().iloc[0],
    'Максимум': columns_of_interest.max(),
    'Минимум': columns_of_interest.min()
}

statistics_df = pd.DataFrame(statistics)
print(statistics_df)

plt.figure(figsize=(15, 10))

for i, column in enumerate(columns_of_interest.columns):
    x = np.array(range(len(data))).reshape(-1, 1)  # Индексы как переменная x
    y = columns_of_interest[column].values  # Температура как переменная y

    model = LinearRegression()
    model.fit(x, y)

    y_pred = model.predict(x)

    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"Уравнение линии тренда для {column}: y = {slope:.2f}x + {intercept:.2f}")

    plt.subplot(3, 1, i + 1)
    plt.scatter(x, y, label='Данные', color='blue')
    plt.plot(x, y_pred, label='Линия тренда', color='red')
    plt.title(f'Линия тренда для {column}')
    plt.xlabel('Время (индексы)')
    plt.ylabel('Температура (°C)')
    plt.legend()

plt.tight_layout()
plt.show()