{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPm0jyF+C5m/TdZ++jjtzm3",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/kateksgva/kate.ksgva.hse/blob/main/bot.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Bz2xXuLs8MKw",
        "outputId": "4019ef59-40c8-4b7e-f5d4-ab3e6b520e53"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/612.8 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[91m━━━━━━━━━━\u001b[0m\u001b[90m╺\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m153.6/612.8 kB\u001b[0m \u001b[31m4.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[90m╺\u001b[0m \u001b[32m604.2/612.8 kB\u001b[0m \u001b[31m9.2 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m612.8/612.8 kB\u001b[0m \u001b[31m7.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h"
          ]
        }
      ],
      "source": [
        "!pip install aiogram -q"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from aiogram import Bot, Dispatcher, types\n",
        "from aiogram.types import ReplyKeyboardMarkup, KeyboardButton\n",
        "from aiogram.filters import Command\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "import gensim\n",
        "from gensim.models import Word2Vec\n",
        "import json\n",
        "import numpy as np\n",
        "import asyncio\n",
        "import sys"
      ],
      "metadata": {
        "id": "Gcjn6dNT8QaZ"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "API_TOKEN = \"7768784827:AAFAzx5WGrc923ZVDt71y7YJbrMAV8NwPhk\"\n",
        "bot = Bot(token=API_TOKEN)\n",
        "dp = Dispatcher()"
      ],
      "metadata": {
        "id": "OznFW4tN8TTF"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Загрузка FAQ из JSON\n",
        "!wget https://raw.githubusercontent.com/vifirsanova/compling/main/tasks/task3/faq.json\n",
        "with open('faq.json', encoding='utf-8') as f:\n",
        "    data = json.load(f)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sxElu5RJ8UQY",
        "outputId": "04ca790c-7054-4b6e-e65c-201ea33b9396"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "--2025-03-04 20:55:42--  https://raw.githubusercontent.com/vifirsanova/compling/main/tasks/task3/faq.json\n",
            "Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...\n",
            "Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.\n",
            "HTTP request sent, awaiting response... 200 OK\n",
            "Length: 2876 (2.8K) [text/plain]\n",
            "Saving to: ‘faq.json’\n",
            "\n",
            "\rfaq.json              0%[                    ]       0  --.-KB/s               \rfaq.json            100%[===================>]   2.81K  --.-KB/s    in 0s      \n",
            "\n",
            "2025-03-04 20:55:42 (53.9 MB/s) - ‘faq.json’ saved [2876/2876]\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Извлечение вопросов и ответов\n",
        "faq_questions = []\n",
        "faq_anwsers = []\n",
        "for q in data.values():\n",
        "    for y in q:\n",
        "        faq_questions.append(y['question'])\n",
        "        faq_anwsers.append(y['answer'])"
      ],
      "metadata": {
        "id": "Y7s92sjU8W3J"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# TF-IDF преобразование\n",
        "vectorizer = TfidfVectorizer()\n",
        "tfidf_matrix = vectorizer.fit_transform(faq_questions)\n",
        "\n",
        "# Word2Vec\n",
        "sentences = [q.split() for q in faq_questions]\n",
        "model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)"
      ],
      "metadata": {
        "id": "S15VB4w-8lg5"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Функция для усреднения векторов слов в вопросе\n",
        "def sentence_vector(sentence, model):\n",
        "    words = sentence.split()\n",
        "    vectors = [model.wv[word] for word in words if word in model.wv]\n",
        "    return np.mean(vectors, axis=0)\n",
        "\n",
        "# Векторизуем вопросы\n",
        "faq_vectors = np.array([sentence_vector(q, model) for q in faq_questions])"
      ],
      "metadata": {
        "id": "Up97fX7e8uRH"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Функция для выбора наиболее подходящего ответа на основе TF-IDF\n",
        "def first_answer(question):\n",
        "    question_vec = vectorizer.transform([question])\n",
        "    similarities = cosine_similarity(question_vec, tfidf_matrix)\n",
        "    best_match_idx = similarities.argmax()\n",
        "    return faq_answers[best_match_idx]"
      ],
      "metadata": {
        "id": "OlLUQ2zg8wIa"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Функция для выбора наиболее подходящего ответа на основе Word2Vec\n",
        "def second_answer(question):\n",
        "    question_vector = sentence_vector(question, model).reshape(1, -1)\n",
        "    similarities = cosine_similarity(question_vector, faq_vectors)\n",
        "    best_match_idx = similarities.argmax()\n",
        "    return faq_answers[best_match_idx]"
      ],
      "metadata": {
        "id": "1Kf1ZhLx8yBB"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Обработка команды /start\n",
        "@dp.message(Command(\"start\"))\n",
        "async def start_command(message: types.Message):\n",
        "    keyboard = ReplyKeyboardMarkup(\n",
        "        keyboard=[\n",
        "            [KeyboardButton(text=\"О компании\")],\n",
        "            [KeyboardButton(text=\"Пожаловаться\")]\n",
        "        ],\n",
        "        resize_keyboard=True\n",
        "    )\n",
        "    await message.answer(\"Привет! Я бот, с радостью отвечу на Ваши вопросы.\", reply_markup=keyboard)"
      ],
      "metadata": {
        "id": "0AD8wJc18ypY"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Обработка кнопки \"О компании\"\n",
        "@dp.message(lambda message: message.text == 'О компании')\n",
        "async def about_company(message: types.Message):\n",
        "    await message.answer(\"Наша компания занимается доставкой товаров по всей стране.\")"
      ],
      "metadata": {
        "id": "tdr3yEyA9Etv"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Обработка кнопки \"Пожаловаться\"\n",
        "@dp.message(lambda message: message.text == \"Пожаловаться\")\n",
        "async def complain(message: types.Message):\n",
        "    await message.answer(\"Пожалуйста, отправьте картинку или скриншот с проблемой.\")"
      ],
      "metadata": {
        "id": "VBH_yM2b9LCh"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Обработка изображений\n",
        "@dp.message(lambda message: message.content_type == \"photo\")\n",
        "async def get_photo(message: types.Message):\n",
        "    file_id = message.photo[-1].file_id\n",
        "    file = await bot.get_file(file_id)\n",
        "    filename = file.file_path.split(\"/\")[-1]\n",
        "    filesize = message.photo[0].file_size\n",
        "    await message.answer(f'Ваш запрос передан специалисту. Название файла: {filename}, размер: {filesize} байт')"
      ],
      "metadata": {
        "id": "ocy-NZPw9a7p"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Обработка вопросов пользователей\n",
        "@dp.message()\n",
        "async def answers(message: types.Message):\n",
        "    question = message.text\n",
        "    # Ответ на основе TF-IDF\n",
        "    tfidf_answer = first_answer(question)\n",
        "    # Ответ на основе Word2Vec\n",
        "    word2vec_answer = second_answer(question)\n",
        "    # Отправляем оба ответа пользователю\n",
        "    await message.answer(f\"Ответ 1: {tfidf_answer}\")\n",
        "    await message.answer(f\"Ответ 2: {word2vec_answer}\")"
      ],
      "metadata": {
        "id": "ZYo5LTn29j5L"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Основной цикл\n",
        "async def main():\n",
        "    await dp.start_polling(bot)\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    await main()\n"
      ],
      "metadata": {
        "id": "dIQxuVHW9mnK"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}