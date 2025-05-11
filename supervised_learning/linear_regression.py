# supervised_learning/linear_regression.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Путь для сохранения модели
MODEL_PATH = 'models/linear_regression_model.pkl'


def train_hydration_model():
    """Обучает модель линейной регрессии для предсказания увлажненности кожи."""
    try:
        data = pd.read_csv('data/skin_data.txt')
        X = data[['oiliness']]
        y = data['moisture']

        model = LinearRegression()
        model.fit(X, y)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        return model

    except FileNotFoundError:
        raise Exception("Файл данных не найден. Проверьте путь 'data/skin_data.csv'.")
    except KeyError as e:
        raise Exception(f"Отсутствует необходимый столбец: {e}")


def predict_hydration(oiliness_level):
    """
    Предсказывает уровень увлажненности кожи на основе жирности.

    Parameters:
    oiliness_level (float): Уровень жирности кожи (0-1)

    Returns:
    float: Прогноз увлажненности (0-1)
    """
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = train_hydration_model()

    if not 0 <= oiliness_level <= 1:
        raise ValueError("Уровень жирности должен быть от 0 до 1.")

    prediction = model.predict([[oiliness_level]])[0]
    return round(float(prediction), 2)
