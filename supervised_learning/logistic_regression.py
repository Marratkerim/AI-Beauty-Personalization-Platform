# supervised_learning/logistic_regression.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Путь для сохранения модели
MODEL_PATH = 'models/logistic_regression_model.pkl'


def train_skin_classification_model():
    """Обучает модель логистической регрессии для классификации типа кожи."""
    try:
        data = pd.read_csv('data/skin_data.txt')
        X = data[['oiliness', 'moisture']]
        y = data['skin_type']

        model = LogisticRegression()
        model.fit(X, y)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        return model

    except FileNotFoundError:
        raise Exception("Файл данных не найден. Проверьте путь 'data/skin_data.csv'.")
    except KeyError as e:
        raise Exception(f"Отсутствует необходимый столбец: {e}")


def classify_skin(oiliness_level, moisture_level):
    """
    Классифицирует тип кожи на основе уровней жирности и увлажненности.

    Parameters:
    oiliness_level (float): Уровень жирности кожи (0-1)
    moisture_level (float): Уровень увлажненности кожи (0-1)

    Returns:
    dict: Тип кожи, вероятность предсказания и текстовое описание
    """
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = train_skin_classification_model()

    if not 0 <= oiliness_level <= 1:
        raise ValueError("Уровень жирности должен быть от 0 до 1.")
    if not 0 <= moisture_level <= 1:
        raise ValueError("Уровень увлажненности должен быть от 0 до 1.")

    prediction = model.predict([[oiliness_level, moisture_level]])[0]
    probability = model.predict_proba([[oiliness_level, moisture_level]])[0]

    return {
        'skin_type': int(prediction),
        'probability': round(float(max(probability)), 2),
        'type_name': 'dry' if prediction == 0 else 'oily'
    }
