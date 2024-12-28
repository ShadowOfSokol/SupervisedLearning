import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Дані
data = {
    "Days": [1, 2, 3, 4, 5],
    "Candies": [2, 4, 6, 8, 10]
}

df = pd.DataFrame(data)
print(df)

# Візуліалізація данних
plt.scatter(df["Days"], df["Candies"], color="blue")
plt.title("Графік залежності")
plt.xlabel("Days")
plt.ylabel("Candies")
plt.show()

# Розділення данних
X = df[["Days"]]
Y = df["Candies"]

# Створення та навчання моделі
model = LinearRegression()
model.fit(X, Y)

# Вивід коефіціентів
print("Вага (W):", model.coef_)
print("Зміщення (b):", model.intercept_)

# Прогнозування
Y_pred = model.predict(X)


# Графік з лінією регресії
plt.scatter(X, Y, color="blue")
plt.plot(X, Y_pred, color="red")
plt.title("Лінія регресії")
plt.xlabel("Days")
plt.ylabel("Candies")
plt.show()

# Прогноз
new_value = [[10]]
predicted_candies = model.predict(new_value)
print(f"Прогноз для {new_value[0][0]}", predicted_candies[0])