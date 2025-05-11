import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

MODEL_PATH = 'models/decision_tree_model.pkl'

def train_decision_tree_model():
    """Обучает модель дерева решений для рекомендаций продуктов."""
    try:
        data = pd.read_csv('data/skin_data_with_products.csv', delimiter=';')  # Обратите внимание на разделитель в CSV
        X = data[['oiliness', 'moisture', 'skin_type']]  # Используем все три признака
        y = data['recommended_product']

        model = DecisionTreeClassifier()
        model.fit(X, y)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        return model

    except FileNotFoundError:
        raise Exception("Файл данных не найден. Проверьте путь 'data/skin_data_with_products.csv'.")
    except KeyError as e:
        raise Exception(f"Отсутствует необходимый столбец: {e}")

def recommend_product(oiliness, moisture, skin_type):
    """
    Рекомендует продукт по типу кожи, жирности и влажности.

    Parameters:
    oiliness (float): Жирность кожи
    moisture (float): Влажность кожи
    skin_type (int): Тип кожи (0 — сухая, 1 — жирная)

    Returns:
    str: Название рекомендованного продукта
    """
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = train_decision_tree_model()

    prediction = model.predict([[oiliness, moisture, skin_type]])[0]
    return prediction
