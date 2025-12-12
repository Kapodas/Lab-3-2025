import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from clearml import Dataset, Task
import os

def fetch_weather_data(city="London", years=3):
    """Загрузка исторических данных из Open-Meteo API"""
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * years)
    
    print(f"Загрузка данных для {city} с {start_date.date()} по {end_date.date()}")
    
    # Координаты для разных городов (можно расширить)
    city_coords = {
        "London": (51.5074, -0.1278),
        "Moscow": (55.7558, 37.6176),
        "Berlin": (52.5200, 13.4050),
        "Paris": (48.8566, 2.3522),
    }
    
    lat, lon = city_coords.get(city, (51.5074, -0.1278))
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min", 
            "precipitation_sum",
            "rain_sum",
            "precipitation_hours",
            "weather_code",
            "wind_speed_10m_max",
            "wind_gusts_10m_max"
        ],
        "timezone": "Europe/London" if city == "London" else "auto"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'daily' not in data:
            print(f"Ошибка: нет данных daily в ответе API")
            print(f"Ответ API: {data}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса к API: {e}")
        return None
    
    # Создаем DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(data['daily']['time']),
        'temp_max': data['daily']['temperature_2m_max'],
        'temp_min': data['daily']['temperature_2m_min'],
        'precipitation_sum': data['daily']['precipitation_sum'],
        'rain_sum': data['daily']['rain_sum'],
        'precipitation_hours': data['daily']['precipitation_hours'],
        'weather_code': data['daily']['weather_code'],
        'wind_speed_max': data['daily']['wind_speed_10m_max'],
        'wind_gusts_max': data['daily']['wind_gusts_10m_max']
    })
    
    print(f"Получено {len(df)} записей")
    
    # Целевая переменная: СРЕДНЯЯ ТЕМПЕРАТУРА вместо rain_probability
    df['temp_avg'] = (df['temp_max'] + df['temp_min']) / 2
    
    # Для совместимости со старым кодом, можно создать rain_probability
    df['rain_probability'] = (df['precipitation_sum'] > 0).astype(int)
    
    # Добавляем календарные признаки
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Сезонные признаки (sin/cos)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Лаги для средней температуры (основная целевая переменная)
    for lag in [1, 2, 3, 7, 14]:
        df[f'temp_avg_lag_{lag}'] = df['temp_avg'].shift(lag)
        df[f'temp_max_lag_{lag}'] = df['temp_max'].shift(lag)
        df[f'temp_min_lag_{lag}'] = df['temp_min'].shift(lag)
        df[f'precip_lag_{lag}'] = df['precipitation_sum'].shift(lag)
        # Лаги для дождя (для совместимости)
        df[f'rain_lag_{lag}'] = df['rain_probability'].shift(lag)
    
    # Скользящие средние для температуры
    for window in [3, 7, 14]:
        df[f'temp_avg_avg_{window}d'] = df['temp_avg'].rolling(window).mean()
        df[f'temp_max_avg_{window}d'] = df['temp_max'].rolling(window).mean()
        df[f'temp_min_avg_{window}d'] = df['temp_min'].rolling(window).mean()
        df[f'precip_avg_{window}d'] = df['precipitation_sum'].rolling(window).mean()
        # Для совместимости
        df[f'rain_avg_{window}d'] = df['rain_probability'].rolling(window).mean()
    
    # Удаляем строки с NaN (из-за лагов)
    df = df.dropna().reset_index(drop=True)
    
    print(f"После очистки: {len(df)} строк, {len(df.columns)} столбцов")
    
    # Статистика
    print(f"Статистика температуры:")
    print(f"  Средняя температура: {df['temp_avg'].mean():.1f}°C")
    print(f"  Min: {df['temp_avg'].min():.1f}°C, Max: {df['temp_avg'].max():.1f}°C")
    print(f"  Средние осадки: {df['precipitation_sum'].mean():.2f} мм")
    print(f"  Дождливых дней: {df['rain_probability'].sum()} ({df['rain_probability'].mean()*100:.1f}%)")
    
    return df

def create_clearml_dataset():
    """Создание ClearML Dataset"""
    
    # Создаем задачу для логирования
    task = Task.init(
        project_name='Lab3_Weather_Forecasting',
        task_name='Dataset_Creation_London_Temp',
        task_type=Task.TaskTypes.data_processing,
        reuse_last_task_id=False
    )
    
    print("Создание ClearML Dataset...")
    
    # Создаем датасет
    dataset = Dataset.create(
        dataset_name="London_Weather_Temperature_v1",
        dataset_project="Lab3_Weather_Forecasting",
        parent_datasets=None
    )
    
    # Загружаем данные
    df = fetch_weather_data(city="London", years=3)
    
    if df is None or len(df) == 0:
        print("Ошибка: не удалось загрузить данные")
        return None
    
    # Сохраняем данные в файл
    csv_filename = f"london_weather_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(os.getcwd(), csv_filename)
    
    print(f"Сохранение данных в файл: {csv_path}")
    df.to_csv(csv_path, index=False)
    
    # Добавляем файл в датасет
    print("Добавление файла в датасет...")
    dataset.add_files(csv_path)
    
    # Добавляем описание
    dataset.set_description(f"""
    Исторические погодные данные для Лондона (3 года)
    
    Источник: Open-Meteo Archive API
    Период: {df['date'].min().date()} - {df['date'].max().date()}
    Город: Лондон, Великобритания
    Координаты: 51.5074° N, -0.1278° W
    
    Основная целевая переменная: temp_avg (средняя температура)
    Дополнительная: rain_probability (бинарная, 1 если были осадки)
    
    Признаки:
    - Температурные: средняя, макс, мин температура
    - Календарные: день недели, день года, месяц, год
    - Сезонные: sin/cos преобразования
    - Лаги температуры: 1, 2, 3, 7, 14 дней
    - Скользящие средние температуры: 3, 7, 14 дней
    - Погодные: осадки, скорость ветра, погодный код
    
    Количество записей: {len(df)}
    Размер файла: {os.path.getsize(csv_path) / 1024:.1f} KB
    
    Создано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)
    
    # Добавляем теги
    dataset.add_tags(['weather', 'london', 'temperature', 'regression', 'time-series'])
    
    # Публикуем датасет
    print("Загрузка датасета на сервер...")
    dataset.upload(verbose=True)
    
    print("Финализация датасета...")
    dataset.finalize()
    
    print(f"\n✅ Dataset успешно создан!")
    print(f"   ID: {dataset.id}")
    print(f"   Название: {dataset.name}")
    print(f"   Проект: {dataset.project}")
    print(f"   Версия: {dataset.version}")
    print(f"   URL: http://localhost:8080/datasets/{dataset.id}")
    
    # Логируем информацию в задачу
    task.get_logger().report_text(f"Dataset создан: {dataset.id}")
    
    # Чистим временный файл
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"Временный файл удален: {csv_path}")
    
    return dataset.id

if __name__ == "__main__":
    print("=" * 60)
    print("Создание датасета с температурными данными")
    print("=" * 60)
    
    dataset_id = create_clearml_dataset()
    
    if dataset_id:
        print(f"\n✅ Готово! Dataset ID: {dataset_id}")
        print(f"   Для использования в обучении:")
        print(f"   dataset = Dataset.get(dataset_project='Lab3_Weather_Forecasting', dataset_name='London_Weather_Temperature_v1')")
    else:
        print("\n❌ Ошибка при создании датасета")