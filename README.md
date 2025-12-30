# InBody ML Telegram Bot (база)

Минимальный каркас проекта для предсказания показателей InBody по регулярным данным одного человека.

## Что внутри
- `data/inbody_sample.csv` — пример данных.
- `scripts/train_model.py` — обучение модели и сохранение артефакта.
- `src/ml/model.py` — загрузка модели и предсказание.
- `src/bot/bot.py` — Telegram-бот для предсказаний.

## Быстрый старт
1) Установи зависимости через Poetry:

```bash
poetry install
```

2) Скопируй пример данных и наполни своими:

```bash
cp data/inbody_sample.csv data/inbody.csv
```

3) Обучи модель:

```bash
poetry run train-model --data data/inbody.csv --out artifacts/model.joblib
```

4) Создай `.env` из примера и укажи токен бота:

```bash
cp .env.example .env
```

5) Запусти бота:

```bash
poetry run bot --model artifacts/model.joblib
```

## Формат данных
Одна строка = один замер. Колонки:
- `date` — дата/время замера
- `weight_kg`, `height_cm`, `age`, `sex`, `waist_cm`, `hip_cm`, `steps`, `training_minutes`, `calories` — входные признаки
- `body_fat_pct`, `lean_mass_kg`, `bmr` — цели (то, что хотим предсказывать)

Единицы всегда в названии колонок.

## Как отправлять данные боту
Формат: `ключ=значение` через пробел. Пример:

```
weight_kg=78.5 height_cm=182 age=29 sex=M waist_cm=86 hip_cm=98 steps=8200 training_minutes=60 calories=2500
```

Бот вернет предсказанные значения целевых метрик.
