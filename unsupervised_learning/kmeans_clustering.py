# unsupervised_learning/kmeans_clustering.py
import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os

MODEL_PATH = 'models/kmeans_model.pkl'

def train_kmeans_model():
    """Обучает K-means для сегментации клиентов."""
    try:
        data = pd.read_csv('data/skin_data.txt')
        X = data[['oiliness', 'moisture']]

        model = KMeans(n_clusters=3)
        model.fit(X)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        return model

    except FileNotFoundError:
        raise Exception("Файл данных не найден.")
    except KeyError as e:
        raise Exception(f"Отсутствует необходимый столбец: {e}")

def segment_clients():
    """
    Сегментирует клиентов.

    Returns:
    list: Список меток кластеров
    """
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = train_kmeans_model()

    labels = model.labels_.tolist()
    return labels
