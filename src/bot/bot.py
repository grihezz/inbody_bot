import argparse
import os
from typing import Any

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from src.ml.model import InBodyModel


def parse_kv_message(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value == "":
            result[key] = value
            continue
        if key == "sex":
            result[key] = value.upper()
            continue
        try:
            if "." in value:
                result[key] = float(value)
            else:
                result[key] = int(value)
        except ValueError:
            result[key] = value
    return result


def format_schema(model: InBodyModel) -> str:
    features = ", ".join(model.features)
    targets = ", ".join(model.targets)
    return (
        "Нужно прислать все признаки:\n"
        f"{features}\n\n"
        "Цели предсказания:\n"
        f"{targets}"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    await update.message.reply_text(
        "Привет! Отправь данные в формате key=value через пробел. "
        "Команда /schema покажет ожидаемые признаки."
    )


async def schema(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    model: InBodyModel = context.bot_data["model"]
    await update.message.reply_text(format_schema(model))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    model: InBodyModel = context.bot_data["model"]
    text = update.message.text or ""
    payload = parse_kv_message(text)

    missing = [c for c in model.features if c not in payload]
    if missing:
        await update.message.reply_text(
            "Не хватает признаков: " + ", ".join(missing)
        )
        return

    try:
        preds = model.predict(payload)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return

    lines = ["Предсказание:"]
    for key, value in preds.items():
        lines.append(f"{key} = {value:.2f}")
    await update.message.reply_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to joblib model")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN is not set in environment")

    args = parse_args()
    model = InBodyModel.load(args.model)

    app = ApplicationBuilder().token(token).build()
    app.bot_data["model"] = model

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("schema", schema))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()


if __name__ == "__main__":
    main()
