!pip install aiogram -q

from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models import Word2Vec
import json
import numpy as np
import asyncio
import sys

API_TOKEN = "7768784827:AAFAzx5WGrc923ZVDt71y7YJbrMAV8NwPhk"
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Загрузка FAQ из JSON
!wget https://raw.githubusercontent.com/vifirsanova/compling/main/tasks/task3/faq.json
with open('faq.json', encoding='utf-8') as f:
    data = json.load(f)

# Извлечение вопросов и ответов
faq_questions = []
faq_anwsers = []
for q in data.values():
    for y in q:
        faq_questions.append(y['question'])
        faq_anwsers.append(y['answer'])

# TF-IDF преобразование
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(faq_questions)

# Word2Vec
sentences = [q.split() for q in faq_questions]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Функция для усреднения векторов слов в вопросе
def sentence_vector(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0)

# Векторизуем вопросы
faq_vectors = np.array([sentence_vector(q, model) for q in faq_questions])

# Функция для выбора наиболее подходящего ответа на основе TF-IDF
def first_answer(question):
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, tfidf_matrix)
    best_match_idx = similarities.argmax()
    return faq_answers[best_match_idx]

# Функция для выбора наиболее подходящего ответа на основе Word2Vec
def second_answer(question):
    question_vector = sentence_vector(question, model).reshape(1, -1)
    similarities = cosine_similarity(question_vector, faq_vectors)
    best_match_idx = similarities.argmax()
    return faq_answers[best_match_idx]

# Обработка команды /start
@dp.message(Command("start"))
async def start_command(message: types.Message):
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="О компании")],
            [KeyboardButton(text="Пожаловаться")]
        ],
        resize_keyboard=True
    )
    await message.answer("Привет! Я бот, с радостью отвечу на Ваши вопросы.", reply_markup=keyboard)

# Обработка кнопки "О компании"
@dp.message(lambda message: message.text == 'О компании')
async def about_company(message: types.Message):
    await message.answer("Наша компания занимается доставкой товаров по всей стране.")

# Обработка кнопки "Пожаловаться"
@dp.message(lambda message: message.text == "Пожаловаться")
async def complain(message: types.Message):
    await message.answer("Пожалуйста, отправьте картинку или скриншот с проблемой.")

# Обработка изображений
@dp.message(lambda message: message.content_type == "photo")
async def get_photo(message: types.Message):
    file_id = message.photo[-1].file_id
    file = await bot.get_file(file_id)
    filename = file.file_path.split("/")[-1]
    filesize = message.photo[0].file_size
    await message.answer(f'Ваш запрос передан специалисту. Название файла: {filename}, размер: {filesize} байт')

# Обработка вопросов пользователей
@dp.message()
async def answers(message: types.Message):
    question = message.text
    # Ответ на основе TF-IDF
    tfidf_answer = first_answer(question)
    # Ответ на основе Word2Vec
    word2vec_answer = second_answer(question)
    # Отправляем оба ответа пользователю
    await message.answer(f"Ответ 1: {tfidf_answer}")
    await message.answer(f"Ответ 2: {word2vec_answer}")

# Основной цикл
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    await main()
