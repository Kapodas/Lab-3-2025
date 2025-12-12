FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Создание директории для моделей
RUN mkdir -p models

# Порт
EXPOSE 8000

# Запуск сервиса
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]