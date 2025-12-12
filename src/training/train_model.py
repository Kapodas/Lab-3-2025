import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task, Dataset, OutputModel, Logger
import os
import warnings
from datetime import datetime
import json

# Игнорировать предупреждения
warnings.filterwarnings('ignore')

def setup_task():
    """Настройка задачи ClearML"""
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ ДЛЯ ПРОГНОЗА СРЕДНЕЙ ТЕМПЕРАТУРЫ")
    print("=" * 60)
    
    task = Task.init(
        project_name='Lab3_Weather_Forecasting',
        task_name=f'Temperature_Prediction_Model_{datetime.now().strftime("%Y%m%d_%H%M")}',
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
        tags=['lightgbm', 'regression', 'weather', 'temperature', 'lab3']
    )
    
    return task

def load_dataset(task):
    """Загрузка датасета из ClearML или локального файла"""
    print("\n📥 Загрузка датасета из ClearML...")
    
    try:
        # Попытка загрузки из ClearML
        dataset = Dataset.get(
            dataset_project='Lab3_Weather_Forecasting',
            dataset_name='London_Weather_Temperature_v1',
            only_completed=True,
            alias='weather_data'
        )
        
        dataset_path = dataset.get_local_copy()
        print(f"Датасет загружен в: {dataset_path}")
        
        # Поиск CSV файла
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("CSV файл не найден в датасете")
        
        csv_path = os.path.join(dataset_path, csv_files[0])
        print(f"Загрузка данных из: {csv_path}")
        
        df = pd.read_csv(csv_path)
        dataset_id = dataset.id
        
    except Exception as e:
        print(f"⚠️  Ошибка загрузки из ClearML: {e}")
        print("Пробуем загрузить локальный файл...")
        
        # Резервный вариант
        csv_path = "../data/weather_data.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dataset_id = 'local_file'
        else:
            # Создаем тестовые данные для демонстрации
            print("Создание тестовых данных...")
            df = create_sample_data()
            dataset_id = 'sample_data'
    
    # Преобразование даты
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Данные загружены: {len(df)} строк, {len(df.columns)} столбцов")
    if 'date' in df.columns:
        print(f"📅 Период: {df['date'].min().date()} - {df['date'].max().date()}")
    
    # Логирование информации о датасете
    task.get_logger().report_text(f"Dataset loaded: {dataset_id}, shape: {df.shape}")
    
    return df, dataset_id

def create_sample_data():
    """Создание тестовых данных если основной датасет недоступен"""
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    n_samples = len(dates)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'date': dates,
        'temp_max': np.random.normal(15, 5, n_samples),
        'temp_min': np.random.normal(8, 3, n_samples),
        'precipitation_sum': np.random.exponential(0.5, n_samples),
        'precipitation_hours': np.random.poisson(2, n_samples),
        'weather_code': np.random.randint(0, 10, n_samples),
        'wind_speed_max': np.random.normal(10, 3, n_samples),
        'rain_sum': np.random.exponential(0.3, n_samples)
    })
    
    # Целевая переменная - средняя температура
    df['temp_avg'] = (df['temp_max'] + df['temp_min']) / 2
    
    # Добавляем лаги и скользящие средние для температуры
    for lag in [1, 2, 3, 7, 14]:
        df[f'temp_avg_lag_{lag}'] = df['temp_avg'].shift(lag)
        df[f'temp_max_lag_{lag}'] = df['temp_max'].shift(lag)
        df[f'temp_min_lag_{lag}'] = df['temp_min'].shift(lag)
        df[f'precip_lag_{lag}'] = df['precipitation_sum'].shift(lag)
    
    for window in [3, 7, 14]:
        df[f'temp_avg_avg_{window}d'] = df['temp_avg'].rolling(window).mean()
        df[f'temp_max_avg_{window}d'] = df['temp_max'].rolling(window).mean()
        df[f'temp_min_avg_{window}d'] = df['temp_min'].rolling(window).mean()
        df[f'precip_avg_{window}d'] = df['precipitation_sum'].rolling(window).mean()
    
    # Календарные признаки
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Сезонные признаки
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Удаляем NaN
    df = df.dropna().reset_index(drop=True)
    
    return df

def prepare_features(df, task):
    """Подготовка признаков для прогноза средней температуры"""
    print("\n🔧 Подготовка признаков для прогноза температуры...")
    
    # Исключаем целевые и временные колонки
    exclude_cols = [
        'date', 'temp_avg',  # Целевая переменная
    ]
    
    # Создаем список признаков (все колонки кроме исключенных)
    features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Используем {len(features)} признаков")
    
    X = df[features]
    y = df['temp_avg']
    
    return X, y, features

def split_data_temporal(X, y, test_size=0.15, val_size=0.15):
    """Временное разделение данных для временных рядов"""
    print("\n📊 Временное разделение данных...")
    
    n_samples = len(X)
    train_size = int(n_samples * (1 - test_size - val_size))
    val_size_abs = int(n_samples * val_size)
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    
    X_val = X.iloc[train_size:train_size + val_size_abs]
    y_val = y.iloc[train_size:train_size + val_size_abs]
    
    X_test = X.iloc[train_size + val_size_abs:]
    y_test = y.iloc[train_size + val_size_abs:]
    
    print(f"Train: {len(X_train)} записей ({len(X_train)/n_samples*100:.1f}%)")
    print(f"Val:   {len(X_val)} записей ({len(X_val)/n_samples*100:.1f}%)")
    print(f"Test:  {len(X_test)} записей ({len(X_test)/n_samples*100:.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(X_train, y_train, X_val, y_val, task):
    """Обучение модели LightGBM для регрессии"""
    print("\n⚙️  Настройка параметров модели для регрессии...")
    
    # Параметры для РЕГРЕССИИ
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 200,
        'verbose': -1,
        'random_state': 42,
    }
    
    # Подключаем параметры к задаче ClearML
    params = task.connect(params)
    
    print("🏋️  Обучение модели LightGBM...")
    
    # Создаем датасеты для LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Обучение с обратным вызовом для логирования в ClearML
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=200,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=20),
            log_to_clearml(task)
        ]
    )
    
    print("✅ Обучение завершено!")
    return model

def log_to_clearml(task):
    """Кастомный callback для логирования метрик в ClearML"""
    def _callback(env):
        if env.iteration % 10 == 0:  # Логируем каждые 10 итераций
            for data_name, eval_name, result, _ in env.evaluation_result_list:
                metric_name = f"{data_name}_{eval_name}"
                task.get_logger().report_scalar(
                    title="Training Metrics",
                    series=metric_name,
                    value=result,
                    iteration=env.iteration
                )
    return _callback

def evaluate_model(model, X_test, y_test, task):
    """Оценка регрессионной модели"""
    print("\n📈 Оценка регрессионной модели...")
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Метрики для регрессии
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
        'std_error': np.std(y_test - y_pred)
    }
    
    print("\n📊 Метрики регрессии на тестовой выборке:")
    print(f"  RMSE: {metrics['rmse']:.2f}°C")
    print(f"  MAE: {metrics['mae']:.2f}°C")
    print(f"  R²: {metrics['r2']:.3f}")
    print(f"  MAPE: {metrics['mape']:.1f}%")
    
    # Логирование метрик в ClearML
    logger = Logger.current_logger()
    for name, value in metrics.items():
        logger.report_scalar(
            title='Test Metrics',
            series=name,
            value=value,
            iteration=0
        )
    
    # Для совместимости возвращаем кортеж из 4 элементов
    return y_pred, None, metrics, None  # y_pred_proba и cm не нужны для регрессии

def analyze_feature_importance(model, features, X_test, task):
    """Анализ важности признаков"""
    print("\n📊 Анализ важности признаков...")
    
    importances = model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n🏆 Топ-10 важнейших признаков:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.2f}")
    
    # Логируем таблицу важности признаков
    task.get_logger().report_table(
        title='Feature Importance',
        series='All Features',
        table_plot=importance_df
    )
    
    return importance_df

def create_plots(model, importance_df, X_test, y_test, y_pred, task):
    """Создание графиков для визуализации регрессии"""
    print("\n🎨 Создание графиков...")
    
    # 1. График важности признаков
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance (Gain)')
    plt.title('Top 15 Feature Importances - Temperature Prediction')
    plt.tight_layout()
    task.get_logger().report_matplotlib_figure(
        title='Feature Importance Plot',
        series='Top 15 Features',
        figure=plt,
        iteration=0
    )
    plt.close()
    
    # 2. График предсказаний vs реальных значений
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title('Actual vs Predicted Temperature')
    plt.grid(True, alpha=0.3)
    
    # Добавляем линию идеального прогноза
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.legend()
    plt.tight_layout()
    task.get_logger().report_matplotlib_figure(
        title='Model Performance',
        series='Actual vs Predicted',
        figure=plt,
        iteration=0
    )
    plt.close()
    
    # 3. График ошибок
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error (°C)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    plt.legend()
    plt.tight_layout()
    task.get_logger().report_matplotlib_figure(
        title='Model Errors',
        series='Error Distribution',
        figure=plt,
        iteration=0
    )
    plt.close()

def save_model(model, features, metrics, task):
    """Сохранение модели и артефактов"""
    print("\n💾 Сохранение модели...")
    
    # Создаем директорию для моделей если её нет
    os.makedirs('models', exist_ok=True)
    
    # Сохраняем модель
    model_path_txt = 'models/temperature_model.txt'
    model_path_json = 'models/temperature_model.json'
    
    model.save_model(model_path_txt)
    print(f"✅ Модель сохранена как: {model_path_txt}")
    
    # Сохраняем метаданные модели
    metadata = {
        'features': features,
        'metrics': metrics,
        'created_at': datetime.now().isoformat(),
        'model_type': 'LightGBM_Regressor',
        'version': '1.0.0',
        'target_variable': 'temp_avg',
        'units': 'degrees_celsius'
    }
    
    with open(model_path_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Метаданные сохранены как: {model_path_json}")
    
    # Загружаем артефакты в ClearML
    print("\n📤 Загрузка артефактов в ClearML...")
    task.upload_artifact('trained_model', model_path_txt)
    task.upload_artifact('model_metadata', model_path_json)
    
    return model_path_txt, model_path_json

def register_model_in_clearml(model_path, task, features, metrics, dataset_id):
    """Регистрация модели в ClearML Model Registry"""
    print("\n🏷️  Регистрация модели в Model Registry...")
    
    try:
        # Создаем объект OutputModel для регистрации
        output_model = OutputModel(
            task=task,
            framework="LightGBM",
            name="Temperature_Predictor",
            tags=['production', 'weather', 'regression', 'temperature', 'lab3']  # ИЗМЕНЕНО
        )
        
        # Обновляем веса модели
        output_model.update_weights(
            weights_filename=model_path,
            auto_delete_file=False,
            iteration=0
        )
        
        # Устанавливаем метаданные
        output_model.update_design(
            config_text=json.dumps({
                'features': features,
                'metrics': metrics,
                'dataset_id': dataset_id,
                'task_id': task.id,
                'created': datetime.now().isoformat(),
                'model_type': 'Temperature Regression',
                'target': 'temp_avg'
            }, indent=2)
        )
        
        # Добавляем дополнительные теги
        output_model.set_tags(['v1.0', 'lightgbm', 'london', 'temperature_prediction'])
        
        print(f"✅ Модель успешно зарегистрирована!")
        print(f"   Model ID: {output_model.id}")
        print(f"   Model Name: {output_model.name}")
        print(f"   URL: http://localhost:8080/models/{output_model.id}")
        
        return output_model
        
    except Exception as e:
        print(f"⚠️  Ошибка при регистрации модели: {e}")
        print("Пробуем альтернативный метод регистрации...")
        
        # Альтернативный метод - сохраняем как артефакт задачи
        task.upload_artifact('production_model', model_path, metadata={
            'features': features,
            'metrics': metrics,
            'registered_manually': True
        })
        
        print("✅ Модель сохранена как артефакт задачи")
        return None

def main():
    """Основная функция"""
    try:
        # 1. Настройка задачи ClearML
        task = setup_task()
        
        # 2. Загрузка данных
        df, dataset_id = load_dataset(task)
        
        # 3. Подготовка признаков
        X, y, features = prepare_features(df, task)
        
        # 4. Разделение данных
        X_train, y_train, X_val, y_val, X_test, y_test = split_data_temporal(
            X, y, test_size=0.15, val_size=0.15
        )
        
        # 5. Обучение модели
        model = train_model(X_train, y_train, X_val, y_val, task)
        
        # 6. Оценка модели
        y_pred, y_pred_proba, metrics, cm = evaluate_model(model, X_test, y_test, task)
        
        # 7. Анализ важности признаков
        importance_df = analyze_feature_importance(model, features, X_test, task)
        
        # 8. Создание графиков
        create_plots(model, importance_df, X_test, y_test, y_pred, task)  # Передаем y_pred вместо y_pred_proba
        
        # 9. Сохранение модели
        model_path_txt, model_path_json = save_model(model, features, metrics, task)
        
        # 10. Регистрация модели в ClearML
        registered_model = register_model_in_clearml(
            model_path_txt, task, features, metrics, dataset_id
        )
        
        # 11. Финальный отчет
        print("\n" + "=" * 60)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 60)
        
        print(f"\n📊 Итоговые метрики:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        
        print(f"\n🔗 Ссылки в ClearML:")
        print(f"  Задача: http://localhost:8080/projects/{task.project}/experiments/{task.id}")
        if registered_model:
            print(f"  Модель: http://localhost:8080/models/{registered_model.id}")
        
        print(f"\n📁 Сохраненные файлы:")
        print(f"  Модель: {model_path_txt}")
        print(f"  Метаданные: {model_path_json}")
        
        print(f"\n🎯 Рекомендации:")
        print("  1. Проверьте модель в веб-интерфейсе ClearML")
        print("  2. Обновите API сервис для использования новой модели")
        print("  3. Протестируйте предсказания на новых данных")
        
        return {
            'task': task,
            'model': model,
            'metrics': metrics,
            'model_path': model_path_txt,
            'registered_model': registered_model
        }
        
    except Exception as e:
        print(f"\n❌ Ошибка во время выполнения: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    
    if result:
        print(f"\n🎉 Все этапы выполнены успешно!")
        print(f"Task ID: {result['task'].id}")
    else:
        print("\n⚠️  Выполнение завершилось с ошибками")