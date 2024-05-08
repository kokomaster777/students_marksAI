import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (KeyboardButton, Message, ReplyKeyboardMarkup, ReplyKeyboardRemove, BotCommand, InputFile)
from aiogram.types.input_file import FSInputFile
import logging



API_TOKEN = '6918569469:AAEwARICCMlTzaag-iKYauMWMioG6hGGD6Y'
# Создаем объекты кнопок
button_1 = KeyboardButton(text='Да')
button_2 = KeyboardButton(text='Нет')
button_3 = KeyboardButton(text='Топ-10 студентов')
button_4 = KeyboardButton(text='Успеваемость')
button_5 = KeyboardButton(text='Средние баллы по предметам')
button_6 = KeyboardButton(text='Вопросов больше нет')

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Добавьте функцию для анализа данных с использованием искусственного интеллекта
def analyze_data(data):
    data = data.iloc[:, 31:34]
    data = data.fillna(data.median())
    X = data[['G1', 'G2']].values
    y = data['G3'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("Средняя квадратичная ошибка:", mse)

    data['Predicted_G3'] = model.predict(data[['G1', 'G2']].values)
    return data


file_results = {}

class FileAnalysisResult:
    def __init__(self, user_id, file_extension):
        self.user_id = user_id
        self.file_extension = file_extension
        self.rating = None
        self.excellent_students_count = None

async def set_main_menu(bot: Bot):
    main_menu_commands = [
        BotCommand(command='/start', description='Начать'),
        BotCommand(command='/help', description='Справка по работе бота')]

    await bot.set_my_commands(main_menu_commands)

dp.startup.register(set_main_menu)

@dp.message(CommandStart())
async def process_start_command(message: Message):
    await message.answer('Привет! \nЯ Телеграм-бот, который спрогнозирует оценки за итоговый экзамен. \nЧтобы я мог это сделать, пришли мне Excel-файл c информацией.')

@dp.message(Command(commands='help'))
async def process_help_command(message: Message):
    await message.answer('Правила пользования ботом:\nЧтобы начать нажми кнопку /start\nЗатем отправь файл excel или csv боту, он проанализирует его.\nПотом он отправит тебе новый файл со столбцом прогнозируемых оценок и предложит тебе узнать дополнительную информацию.')



@dp.message(F.document)
async def handle_document(message: types.Message):
    document = message.document
    if document.file_name.endswith('.xlsx') or document.file_name.endswith('.xls') or document.file_name.endswith('.csv'):
        file_id = document.file_id
        file_info = await bot.get_file(file_id)
        file_url = file_info.file_path

        file_extension = document.file_name.split('.')[-1]

        file = await bot.download_file(file_url)
        if file_extension in ['xlsx', 'xls']:
            data = pd.read_excel(file)  # Чтение данных из Excel-файла
        elif file_extension == 'csv':
            data = pd.read_csv(file)  # Чтение данных из CSV-файла
        else:
            await message.answer('Пожалуйста, пришлите Excel-файл или CSV-файл.')
            return

        file_results[message.from_user.id] = FileAnalysisResult(message.from_user.id, file_extension)

        await message.answer(f"{file_extension.upper()} файл успешно загружен.")

        analyzed_data = analyze_data(data)  # Анализ данных с использованием искусственного интеллекта

        # Сохраните анализированные данные в новый Excel-файл
        analyzed_data.to_excel("analyzed_data.xlsx", index=False)
        print("Sending analyzed data...")
        user_id = message.from_user.id
        document = FSInputFile('analyzed_data.xlsx')
        await bot.send_document(user_id, document)
        keyboard_additional = ReplyKeyboardMarkup(
            keyboard=[
                [button_1, button_2],
            ],
            resize_keyboard=True
        )
        await message.answer(
            text='Хочешь узнать дополнительную информацию об учениках?', reply_markup=keyboard_additional
        )
    else:
        await message.answer('Пожалуйста, пришлите Excel-файл или CSV-файл.')


@dp.message(F.text == 'Да')
async def process_answer_yes(message: Message):
    keyboard_additional = ReplyKeyboardMarkup(
        keyboard=[
            [button_3, button_4],
            [button_5, button_6]
        ],
        resize_keyboard=True
    )
    await message.answer(
        text='Отлично! Тогда выбирай вопрос, который тебя интересует?', reply_markup=keyboard_additional
    )

@dp.message(F.text == 'Нет')
async def process_answer_no(message: Message):
    await message.answer(
        text='Хорошо. Если появились вопросы по работе бота, то нажми на кнопку в меню - Справка по работе бота.',
        reply_markup=ReplyKeyboardRemove()
    )

@dp.message(F.text == 'Топ-10 студентов')
async def process_answer_raiting(message: Message):
    user_id = message.from_user.id
    result = file_results.get(user_id)
    if result:
        await message.answer("Бот еще в разработке, ответ на ваш вопрос появится позже")
    else:
        await message.answer("Для вашего пользователя нет результатов анализа файла.")

@dp.message(F.text == 'Успеваемость')
async def process_answer_raiting(message: Message):
    user_id = message.from_user.id
    result = file_results.get(user_id)
    if result:
        await message.answer("Бот еще в разработке, ответ на ваш вопрос появится позже")
    else:
        await message.answer("Для вашего пользователя нет результатов анализа файла.")

@dp.message(F.text == 'Средние баллы по предметам')
async def process_answer_raiting(message: Message):
    user_id = message.from_user.id
    result = file_results.get(user_id)
    if result:
        await message.answer("Бот еще в разработке, ответ на ваш вопрос появится позже")
    else:
        await message.answer("Для вашего пользователя нет результатов анализа файла.")

@dp.message(F.text == 'Вопросов больше нет')
async def process_answer_raiting(message: Message):
    await message.answer(
        text='Хорошо. Если появились вопросы по работе бота, то нажми на кнопку в меню - Справка по работе бота.',
        reply_markup=ReplyKeyboardRemove()
    )


if __name__ == '__main__':
    dp.run_polling(bot)