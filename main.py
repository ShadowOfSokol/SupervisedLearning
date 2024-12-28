# Залежність годин за комп'ютером та енергії витраченої
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Генеруємо дані
np.random.seed(24)
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
energy_usage = 10 * hours + 15 + np.random.normal(0, 5, len(hours))  # Додаємо шум

# Вхідні та вихідні дані
X = hours.reshape(-1, 1)  # Перетворюємо в 2D-масив
Y = energy_usage  # Вихідні дані

# Створення та навчання моделі
model = LinearRegression()
model.fit(X, Y)

# Вивід коефіціентів
print("Вага (W):", model.coef_)
print("Зміщення (b):", model.intercept_)

# Прогнозування
Y_pred = model.predict(X)


# Графік з лінією регресії
plt.scatter(X, Y, color="blue", label="Ральні дані")
plt.plot(X, Y_pred, color="red", label="Лінійна регресія")
plt.title("Лінія регресії: Витрати енергії від годин роботи")
plt.xlabel("Години роботи")
plt.ylabel("Витрати енергії")
plt.show()

# Прогноз
new_value = [[10]]
predicted_candies = model.predict(new_value)
print(f"Прогноз для {new_value[0][0]}", predicted_candies[0])