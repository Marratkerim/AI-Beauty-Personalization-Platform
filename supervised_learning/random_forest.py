# supervised_learning/random_forest.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

MODEL_PATH = 'models/random_forest_model.pkl'

def train_random_forest_model():
    """Обучает модель случайного леса для мультиклассовой классификации кожи."""
    try:
        data = pd.read_csv('data/skin_data.txt')
        X = data[['oiliness', 'moisture']]
        y = data['skin_type']

        model = RandomForestClassifier()
        model.fit(X, y)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        return model

    except FileNotFoundError:
        raise Exception("Файл данных не найден.")
    except KeyError as e:
        raise Exception(f"Отсутствует необходимый столбец: {e}")

def multi_classify_skin(oiliness_level, moisture_level):
    """
    Классифицирует тип кожи с помощью случайного леса.

    Returns:
    int: Тип кожи
    """
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = train_random_forest_model()

    prediction = model.predict([[oiliness_level, moisture_level]])[0]
    return int(prediction)
