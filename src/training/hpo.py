import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import os
from clearml import Task, Dataset, Logger

def main():
    print("=" * 60)
    print("HPO ДЛЯ РЕГРЕССИИ (предсказание температуры)")
    print("=" * 60)
    
    # Создаем одну задачу для всего HPO
    task = Task.init(
        project_name='Lab3_Weather_Forecasting',
        task_name='HPO_Regression_temp_avg',
        task_type='optimizer'
    )
    
    logger = Logger.current_logger()
    
    # Загрузка данных
    print("📥 Загрузка данных...")
    
    dataset = Dataset.get(
        dataset_project='Lab3_Weather_Forecasting',
        dataset_name='London_Weather_Temperature_v1',
        only_completed=True,
        alias='weather_data_regression'
    )
    
    dataset_path = dataset.get_local_copy()
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    csv_path = os.path.join(dataset_path, csv_files[0])
    
    df = pd.read_csv(csv_path)
    
    # Подготовка признаков для регрессии (предсказываем temp_avg)
    exclude_features = [
        'date', 'rain_probability',  # rain_probability исключаем, т.к. это бинарная переменная
        # Для регрессии оставляем большинство признаков
    ]
    
    features = [col for col in df.columns if col not in exclude_features]
    print(f"✅ Используем {len(features)} признаков")
    
    X = df[features]
    y = df['temp_avg']  # Целевая переменная - средняя температура
    
    # Разделение
    n = len(df)
    train_size = int(n * 0.6)
    val_size = int(n * 0.2)
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:train_size+val_size]
    y_val = y.iloc[train_size:train_size+val_size]
    X_test = X.iloc[train_size+val_size:]
    y_test = y.iloc[train_size+val_size:]
    
    def objective(trial):
        """Целевая функция для Optuna - РЕГРЕССИЯ"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'n_estimators': 150,
            'verbose': -1,
            'random_state': 42
        }
        
        # ИСПРАВЛЕНО: Используем LGBMRegressor вместо LGBMClassifier
        model = lgb.LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Для регрессии используем RMSE или MAE как метрику
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        # Логируем метрики в основную задачу
        logger.report_scalar('HPO_Trials', 'rmse', rmse, trial.number)
        
        # Минимизируем RMSE (чем меньше, тем лучше)
        return rmse
    
    # Запускаем оптимизацию (теперь direction='minimize' для RMSE)
    print(f"\n🚀 Запуск 15 испытаний HPO для регрессии...")
    study = optuna.create_study(direction='minimize')  # ИЗМЕНЕНО: minimize т.к. RMSE
    study.optimize(objective, n_trials=15, show_progress_bar=True)
    
    # Результаты
    print(f"\n✅ Лучший RMSE: {study.best_value:.4f}°C")
    print("🎯 Лучшие параметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Обучаем финальную модель с лучшими параметрами
    print("\n🏋️  Обучение финальной регрессионной модели...")
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    # ИСПРАВЛЕНО: Используем LGBMRegressor
    best_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        verbose=-1,
        random_state=42,
        n_estimators=200,
        **study.best_params
    )
    
    best_model.fit(X_train_full, y_train_full)
    
    # Тестирование регрессионной модели
    y_pred_test = best_model.predict(X_test)
    
    # Метрики для регрессии
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n📈 Результаты на тесте (регрессия):")
    print(f"  RMSE: {test_rmse:.4f}°C")
    print(f"  MAE: {test_mae:.4f}°C")
    print(f"  R²: {test_r2:.4f}")
    
    # Сохраняем модель
    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_hpo_regression.txt'  # ИЗМЕНЕНО имя файла
    best_model.booster_.save_model(model_path)
    
    print(f"\n💾 Модель сохранена: {model_path}")
    
    # Логируем финальные метрики
    logger.report_scalar('Final_Model', 'test_rmse', test_rmse, 0)
    logger.report_scalar('Final_Model', 'test_mae', test_mae, 0)
    logger.report_scalar('Final_Model', 'test_r2', test_r2, 0)
    
    # Сохраняем информацию о признаках для использования в API
    feature_info = {
        'features': features,
        'target': 'temp_avg',
        'best_params': study.best_params,
        'test_metrics': {
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'r2': float(test_r2)
        }
    }
    
    import json
    with open('models/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"\n💾 Информация о признаках сохранена: models/feature_info.json")
    
    print(f"\n🔗 Task URL: http://localhost:8080/projects/{task.project}/experiments/{task.id}")
    
    return study.best_params, test_rmse

if __name__ == "__main__":
    best_params, test_score = main()
    print(f"\n🎉 Готово! Регрессионная модель сохранена в models/best_hpo_regression.txt")