# supervised_learning/knn.py
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

MODEL_PATH = 'models/knn_model.pkl'

def train_knn_model():
    """Обучает модель K-ближайших соседей для поиска похожих клиентов."""
    try:
        data = pd.read_csv('data/skin_data_with_text_groups')
        X = data[['oiliness', 'moisture']]
        y = data['client_group']

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        return model

    except FileNotFoundError:
        raise Exception("Файл данных не найден.")
    except KeyError as e:
        raise Exception(f"Отсутствует необходимый столбец: {e}")

def find_similar_clients(oiliness_level, moisture_level):
    """
    Находит группу похожих клиентов.

    Returns:
    int: Номер группы клиентов
    """
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = train_knn_model()

    prediction = model.predict([[oiliness_level, moisture_level]])[0]
    return int(prediction)
