# supervised_learning/svm.py
import pandas as pd
from sklearn.svm import SVC
import joblib
import os

MODEL_PATH = 'models/svm_model.pkl'

def train_svm_model():
    """Обучает SVM для точной классификации типа кожи."""
    try:
        data = pd.read_csv('data/skin_data.txt')
        X = data[['oiliness', 'moisture']]
        y = data['skin_type']

        model = SVC(probability=True)
        model.fit(X, y)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        return model

    except FileNotFoundError:
        raise Exception("Файл данных не найден.")
    except KeyError as e:
        raise Exception(f"Отсутствует необходимый столбец: {e}")

def precise_classify_skin(oiliness_level, moisture_level):
    """
    Точная классификация типа кожи через SVM.

    Returns:
    dict: Предсказанный тип кожи и вероятность
    """
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = train_svm_model()

    prediction = model.predict([[oiliness_level, moisture_level]])[0]
    probability = model.predict_proba([[oiliness_level, moisture_level]])[0]

    return {
        'skin_type': int(prediction),
        'probability': round(float(max(probability)), 2)
    }
