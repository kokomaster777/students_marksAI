# Импорт необходимых библиотек
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data_original = pd.read_csv('data_original.csv', sep=';', quotechar='"')

# Пропущенные значения
data_original.isnull().sum()

data_original = data_original[data_original['Тип оценки'] == 'итоговая']
data_original = data_original[data_original['Уровень подготовки'] == 'Бакалавр']
desired_values = ['Прикладная информатика	', 'Программная инженерия', 'Информатика и вычислительная техника']
data_original = data_original[data_original['Наименование напр. подготовки'].isin(desired_values)]

data_original = data_original.dropna(subset=['Балл'])

# Определяем столбец, который не должен быть уникальным
specific_column = 'Unnamed: 0'

# Создаем маску дубликатов по всем столбцам, кроме 'Unnamed: 0'
duplicates_mask = data_original.duplicated(subset=data_original.columns.difference([specific_column]))

# Удаляем строки, для которых дубликаты обнаружены
data_no_duplicates = data_original[~duplicates_mask]

# Извлечение нужных столбцов
data = data_no_duplicates[["UUID студента", "Наименование дисциплины", "Оценка", "Балл"]]
# Заменяем запятую на точку в столбце "Балл" и преобразуем значения в числовой формат
data.loc[:, 'Балл'] = data['Балл'].str.replace(',', '.').astype(float)

df = data

# Преобразование таблицы
transformed_df = df.pivot_table(index='UUID студента', columns='Наименование дисциплины', values='Балл').reset_index()

# Удаление имени столбцов (дисциплин)
transformed_df.columns.name = None

df = transformed_df

# Удаление строк с пропущенными значениями оценки за выпускную работу
df = df.dropna(subset=['Выполнение и защита выпускной квалификационной работы'])

# Создаем матрицу признаков X и вектор целевой переменной y
X = df.drop(columns=['UUID студента', 'Выполнение и защита выпускной квалификационной работы'])
y = df['Выполнение и защита выпускной квалификационной работы']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем пайплайн для обработки данных и обучения модели
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', LinearRegression())
])

# Обучение модели
pipeline.fit(X_train, y_train)

# Предсказание значений для тестового набора
y_pred = pipeline.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
print("Средняя квадратичная ошибка:", mse)
