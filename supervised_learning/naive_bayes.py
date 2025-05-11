# supervised_learning/naive_bayes.py
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os

MODEL_PATH = 'models/naive_bayes_model.pkl'
VECTORIZER_PATH = 'models/naive_bayes_vectorizer.pkl'

def train_review_sentiment_model():
    """Обучает наивный байесовский классификатор для анализа отзывов."""
    try:
        data = pd.read_csv('data/reviews.txt')
        X = data['review']
        y = data['sentiment']

        vectorizer = CountVectorizer()
        X_vec = vectorizer.fit_transform(X)

        model = MultinomialNB()
        model.fit(X_vec, y)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        return model, vectorizer

    except FileNotFoundError:
        raise Exception("Файл данных не найден.")
    except KeyError as e:
        raise Exception(f"Отсутствует необходимый столбец: {e}")

def analyze_review(review_text):
    """
    Анализирует отзыв клиента.

    Returns:
    str: 'positive' или 'negative'
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        model, vectorizer = train_review_sentiment_model()

    review_vec = vectorizer.transform([review_text])
    prediction = model.predict(review_vec)[0]
    return prediction
