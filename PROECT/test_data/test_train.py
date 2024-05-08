import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = pd.read_csv('train.csv')

# Пропущенные значения
data.isnull().sum()

data = data.iloc[:, 31:34]
# Отобразим результат
data

# Заполним пустые значения в каждом столбце медианой столбца
data = data.fillna(data.median())
data

data.isnull().sum()

# Создаем матрицу признаков X и вектор целевой переменной y
X = data[['G1', 'G2']].values
y = data['G3'].values

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание значений для тестового набора
y_pred = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
print("Средняя квадратичная ошибка:", mse)
