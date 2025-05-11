import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

MODEL_PATH = 'models/gradient_boosting_model.pkl'
DATA_PATH = 'data/skin_data_with_products.csv'

def train_gradient_boosting_model():
    """Обучает градиентный бустинг для прогноза эффективности продукта."""
    try:
        data = pd.read_csv('data/skin_data_with_products.csv', delimiter=';')
        X = data[['oiliness', 'moisture']]
        y = data['product_effectiveness']

        model = GradientBoostingRegressor()
        model.fit(X, y)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        return model

    except FileNotFoundError:
        raise Exception("Файл данных не найден.")
    except KeyError as e:
        raise Exception(f"Отсутствует необходимый столбец: {e}")


def predict_product_effectiveness(oiliness_level, moisture_level):
    """
    Предсказывает эффективность продукта по уровням oiliness и moisture.

    Args:
    oiliness_level (float): Уровень жирности кожи (0–1).
    moisture_level (float): Уровень увлажненности кожи (0–1).

    Returns:
    float: Прогнозируемая эффективность продукта (0–1).
    """
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            model = train_gradient_boosting_model()

        # Модель ожидает 2D массив
        prediction = model.predict([[oiliness_level, moisture_level]])[0]
        return round(float(prediction), 2)

    except Exception as e:
        raise Exception(f"Ошибка при предсказании: {str(e)}")
