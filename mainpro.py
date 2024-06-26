import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import uniform, randint
from sklearn.metrics import mean_squared_error, r2_score

import io
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (KeyboardButton, ReplyKeyboardMarkup, Message, BotCommand)
from aiogram.types.input_file import FSInputFile
import logging
import aiohttp
import re
import os



API_TOKEN = '7048524483:AAHmfqEbZ05x2555Vn8woMwtfgc7Rxvozls'

# Создаем объекты кнопок

button_1 = KeyboardButton(text='Количество студентов')
button_2 = KeyboardButton(text='Нет')
button_3 = KeyboardButton(text='Топ-10 студентов')
button_4 = KeyboardButton(text='Категории оценок')
button_5 = KeyboardButton(text='Средние баллы по предметам')
button_6 = KeyboardButton(text='Вопросов больше нет')

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Добавьте функцию для анализа данных с использованием искусственного интеллекта
def analyze_data(data):
    def mean_ignore_zeros(row):
        valid_values = pd.to_numeric(row, errors='coerce')  # Преобразование в числа, игнорируя нечисловые значения
        valid_values = valid_values[valid_values != 0]  # Игнорирование значений 0
        if len(valid_values) == 0:
            return np.nan  # Возвращаем NaN, если все значения равны нулю или нечисловые
        return valid_values.mean()

    data = data[data['Тип оценки'] == 'итоговая']
    data = data[data['Уровень подготовки'] == 'Бакалавр']
    desired_values = ['Прикладная информатика', 'Программная инженерия', 'Информатика и вычислительная техника']
    data = data[data['Наименование напр. подготовки'].isin(desired_values)]

    data = data.dropna(subset=['Балл'])

    # Определяем столбец, который не должен быть уникальным
    specific_column = 'Unnamed: 0'

    # Создаем маску дубликатов по всем столбцам, кроме 'Unnamed: 0'
    duplicates_mask = data.duplicated(subset=data.columns.difference([specific_column]))

    # Удаляем строки, для которых дубликаты обнаружены
    data_no_duplicates = data[~duplicates_mask]

    data_no_duplicates = data_no_duplicates[
        ["UUID студента", "Форма освоения", "Пол", "Наименование дисциплины", "Балл"]]

    data_no_duplicates.loc[:, 'Балл'] = data_no_duplicates['Балл'].str.replace(',', '.').astype(float)

    # Не учитываем пол
    data_no_duplicates = data_no_duplicates[["UUID студента", "Форма освоения", "Наименование дисциплины", "Балл"]]

    # преобразование таблицы
    pivot_df = data_no_duplicates.pivot_table(index=['UUID студента', 'Форма освоения'],
                                              columns='Наименование дисциплины', values='Балл')

    # Сброс индексов для получения DataFrame
    pivot_df = pivot_df.reset_index()

    # Заполнение NaN значений нулями (если необходимо)
    pivot_df = pivot_df.fillna(0)



    # Переименование столбцов для удобства (удаление уровня столбцов)
    pivot_df.columns.name = None  # Удаление имени уровня столбцов
    pivot_df.columns = ['UUID студента', 'Форма освоения'] + [col for col in pivot_df.columns if
                                                              col not in ['UUID студента', 'Форма освоения']]
    df_cleaned = pivot_df.drop(columns='Выполнение и защита выпускной квалификационной работы')


    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]

    df_cleaned = df_cleaned[['UUID студента', 'Форма освоения', 'Культура делового письма', 'Организационный менеджмент', 'Цифровая обработка изображений',
                             'Средства управления информационными ресурсами автоматизированных систем', 'Правоведение',
                             'Теория кодирования', 'Построение масштабируемых сетей', 'Социология',
                             'Основы электроники / Arduino', 'Проектный интенсив 1-ВС', 'Основы программной инженерии',
                             'Экономика', 'Корпоративные информационные системы', 'Проектный интенсив 2-ВС',
                             'Безопасность операционных систем', 'Интеллектуальные системы и представление знаний',
                             'Компьютерная графика', 'Технология командной разработки программного обеспечения',
                             'Проектный интенсив 3-ВС', 'Введение в разработку игр',
                             'Прикладное программирование на C\\C++', 'Прикладное программирование на PHP',
                             'Критическое мышление', 'Программирование на PHP', 'Программирование на C\\C++',
                             'Основы игровых механик', 'Искусство ведения переговоров', 'Основы дизайна',
                             'Прикладное программное обеспечение', 'Интернет-маркетинг в бизнесе', 'Психология',
                             'Прикладное программирование на Java', 'Прикладное программирование на Python',
                             'Программирование на Java', 'Системная инженерия', 'Трехмерная визуализация',
                             'Самоменеджмент', 'Информационная безопасность', 'Физическая культура', 'Математика',
                             'Учебная практика, ознакомительная', 'Производственная практика, преддипломная',
                             'Web-технологии', 'Управление проектами', 'Алгебра и геометрия', 'Проектный практикум 3-А',
                             'Теория вероятностей и математическая статистика', 'Компьютерные сети',
                             'Профессиональный курс. Спецкурс 3', 'Моделирование сложных процессов и систем',
                             'Программирование', 'Математические методы для разработчиков 3',
                             'Виртуализация и облачные технологии', 'Проектный практикум 2-А',
                             'Профессиональный курс. Спецкурс 2', 'Майнор 1',
                             'Дискретная математика и математическая логика', 'Введение в специальность',
                             'Проектирование информационных систем', 'Производственная практика, технологическая',
                             'Проектный практикум 4-А', 'Технологии программирования',
                             'Математические методы для разработчиков 1', 'Проектный практикум 5-А',
                             'Основы проектной деятельности', 'Дискретные структуры данных', 'Иностранный язык',
                             'Информационные технологии и сервисы', 'Физика',
                             'Экономические и гуманитарные аспекты информационных технологий 1',
                             'Проектный практикум 1-А', 'Алгоритмы и анализ сложности',
                             'Безопасность жизнедеятельности']]
    transformed_df = df_cleaned
    transformed_df.columns.name = None
    transformed_df.fillna(0, inplace=True)

    transformed_df['Средний балл'] = transformed_df.iloc[:, 1:].apply(mean_ignore_zeros, axis=1)

    '''
    df = df_cleaned.drop(columns='UUID студента')

    # Определение целевой переменной и признаков
    target = 'Выполнение и защита выпускной квалификационной работы'
    X = df.drop(columns=[target])
    y = df[target]

    # Определение числовых и категориальных признаков
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Преобразователь для числовых и категориальных признаков
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='constant', fill_value=0), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

    # Полный пайплайн
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(random_state=42, n_jobs=-1))
    ])

    # Определение параметров для поиска
    param_distributions = {
        'model__learning_rate': uniform(0.01, 0.3),
        'model__n_estimators': randint(100, 200),
        'model__max_depth': randint(3, 12),
        'model__min_child_weight': randint(1, 12),
        'model__subsample': uniform(0.6, 0.4),
        'model__colsample_bytree': uniform(0.6, 0.4),
        'model__reg_alpha': uniform(0, 1),
        'model__reg_lambda': uniform(0, 1)
    }

    # Использование RandomizedSearchCV для настройки гиперпараметров
    random_search = RandomizedSearchCV(
        pipeline, param_distributions, n_iter=200, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
    )

    # Обработка целевой переменной
    y_train = y.fillna(0)

    # Обучение модели
    random_search.fit(X, y_train)

    # Лучшая модель после настройки гиперпараметров
    best_model = random_search.best_estimator_

    # Прогнозирование на обучающей выборке
    y_pred = best_model.predict(X)

    # Оценка модели
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    df['UUID студента'] = df_cleaned['UUID студента']
    df['Прогнозируемая оценка'] = best_model.predict(X).round().astype(int)
    df = df[['UUID студента', 'Выполнение и защита выпускной квалификационной работы', 'Прогнозируемая оценка']]
    print(f"Средняя квадратичная ошибка: {mse}")
    print(f"Коэффициент детерминации R²: {r2 * 100:.2f}%")
    '''

    best_model = joblib.load('best_model.pkl')

    # Применение к новым данным
    new_data_transformed = best_model.named_steps['preprocessor'].transform(df_cleaned)
    predictions = best_model.named_steps['model'].predict(new_data_transformed)

    # Создание таблицы с UUID студента и прогнозируемыми оценками
    results = pd.DataFrame({
        'UUID студента': df_cleaned['UUID студента'],
        'Прогнозируемая оценка': predictions
    })
    return results, transformed_df


def categorize_score(score):
    if score >= 80:
        return 'Отлично'
    elif score >= 60:
        return 'Хорошо'
    elif score >= 40:
        return 'Удовлетворительно'
    else:
        return 'Неудовлетворительно'

def generate_pie_chart(counts, filename):
    plt.figure(figsize=(7, 7))
    counts.plot(kind='pie', autopct='%1.1f%%', colors=['#76c7c0', '#ff9999', '#66b3ff', '#99ff99'], startangle=140)
    plt.title('Количество студентов в каждой категории оценки', fontsize=16, weight='bold')
    plt.ylabel('')  # Убираем подпись оси y для круговой диаграммы
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.close()

async def set_main_menu(bot: Bot):
    main_menu_commands = [
        BotCommand(command='/start', description='Начать'),
        BotCommand(command='/help', description='Справка по работе бота')]

    await bot.set_my_commands(main_menu_commands)

dp.startup.register(set_main_menu)


@dp.message(CommandStart())
async def process_start_command(message: Message):
    await message.answer('Привет! \nЯ Телеграм-бот, который спрогнозирует оценки за итоговый экзамен. \nЧтобы я мог это сделать, загрузи Excel-файл или CSV-файл со всей информацией в Goggle Drive и отправь мне на него ссылку!')
    await message.answer('Столбцы, которые должны присутствовать в вашем файле: \nUUID студента;\nКод направления подготовки;\nНаименование направления подготовки;\nФорма освоения;\nКурс;\nКод образовательной программы;\nОбразовательная программа;\nУровень подготовки;\nПол;\nНаименование дисциплины;\nОценка;\nБалл;\nСеместр;\nСквозной семестр;\nГод;\nТип оценки')

@dp.message(Command(commands='help'))
async def process_help_command(message: Message):
    await message.answer('Правила пользования ботом:\nЧтобы начать нажми кнопку /start\nЗатем отправь файл excel или csv боту, он проанализирует его.\nПотом он отправит тебе новый файл со столбцом прогнозируемых оценок и предложит тебе узнать дополнительную информацию.')

@dp.message(F.text=='Категории оценок')
async def handle_filter_by_category(message: Message):
    user_id = message.from_user.id
    # Сгенерировать имя файла для пользователя
    file_name = f"analyzed_data_{user_id}.xlsx"
    analyzed_data = pd.read_excel(file_name)
    counts = analyzed_data['Категория оценки'].value_counts()
    # Generate and save pie chart
    chart_filename = f"pie_chart_{user_id}.png"
    generate_pie_chart(counts, chart_filename)

    chart_file = FSInputFile(chart_filename)
    # Send saved image file to user
    await message.answer_photo(photo=chart_file, caption=f"Распределение по категориям оценки:\n{counts.to_string()}")

    # Clean up: Delete the saved chart file after sending
    os.remove(chart_filename)

@dp.message(F.text=='Количество студентов')
async def handle_filter_by_category(message: Message):
    user_id = message.from_user.id

    file_name = f"analyzed_data_{user_id}.xlsx"
    analyzed_data = pd.read_excel(file_name)
    await message.answer(f"Количество студентов: {len(analyzed_data)}")

@dp.message(F.text=='Топ-10 студентов')
async def handle_filter_by_category(message: Message):
    user_id = message.from_user.id
    file_name = f"transformed_df_{user_id}.xlsx"
    transformed_df = pd.read_excel(file_name)
    top_10_students = transformed_df.sort_values(by='Средний балл', ascending=False).head(10)
    top_10_students_data = top_10_students[['UUID студента', 'Средний балл']].values

    formatted_output = ""
    for i, (student_id, avg_score) in enumerate(top_10_students_data, 1):
        formatted_output += f"{i}. \nUUID студента:\n{student_id}\nСредний балл:\n{avg_score}\n\n"

    await message.answer(f"Топ 10 студентов по среднему баллу:\n\n{formatted_output}")


def is_google_drive_file_link(text):
    regex = r'https://drive\.google\.com/file/d/([a-zA-Z0-9-_]{33})/view\?.*'
    match = re.search(regex, text)
    return match is not None

@dp.message(F.text)
async def handle_message(message: types.Message):
    if message.text:
        if is_google_drive_file_link(message.text):
            file_url = message.text
            file_id = re.search(r'https://drive\.google\.com/file/d/([a-zA-Z0-9-_]{33})/view\?.*', file_url).group(1)

            # Скачивание файла
            async with aiohttp.ClientSession() as session:
                async with session.get('https://drive.google.com/uc?export=download&id=' + file_id) as response:
                    if response.status == 200:
                        file_extension = response.headers.get('Content-Disposition').split(';')[1].split('=')[1].strip(
                            '"')
                        file_data = {
                            'extension': file_extension,
                            'content': await response.read()  # Read content asynchronously
                        }
                    else:
                        file_data = None

            if file_data:
                if file_data:
                    file_like_object = io.BytesIO(file_data['content'])

                    data = pd.read_csv(file_like_object, sep=';')
                    user_id = message.from_user.id
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    # Создание директории с именем, основанным на текущем времени
                    directory_name = current_time
                    os.makedirs(directory_name, exist_ok=True)

                    # Полные пути для сохранения файлов
                    file_name = os.path.join(directory_name, f"analyzed_data_{user_id}.xlsx")
                    file_name_transformed = os.path.join(directory_name, f'transformed_df_{user_id}.xlsx')

                    analyzed_data, transformed_df = analyze_data(data)
                    analyzed_data['Категория оценки'] = analyzed_data['Прогнозируемая оценка'].apply(categorize_score)

                    transformed_df.to_excel(file_name_transformed, index=False)
                    analyzed_data.to_excel(file_name, index=False)

                    document = FSInputFile(file_name)
                    await bot.send_document(message.from_user.id, document)
                    keyboard_additional = ReplyKeyboardMarkup(
                        keyboard=[
                            [button_1],
                            [button_3, button_4],
                        ],
                        resize_keyboard=True
                    )
                    await message.answer(
                        text='Хочешь узнать дополнительную информацию об учениках?', reply_markup=keyboard_additional
                    )
                else:
                    await message.answer('Пожалуйста, пришлите ссылку на файл Google Drive.')
            else:
                await message.answer('Пожалуйста, пришлите ссылку на файл Google Drive.')

if __name__ == '__main__':
    dp.run_polling(bot)
