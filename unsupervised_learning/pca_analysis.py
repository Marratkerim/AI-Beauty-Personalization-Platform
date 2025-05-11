import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # Для использования без графического интерфейса
import matplotlib.pyplot as plt
import os

def visualize_data():
    """Визуализирует данные через PCA."""
    try:
        # Загрузка данных
        data = pd.read_csv('data/skin_data.txt')
        print(data.head())  # Проверка данных

        # Проверка наличия нужных столбцов
        if 'oiliness' not in data.columns or 'moisture' not in data.columns:
            raise Exception("Необходимые столбцы 'oiliness' и 'moisture' отсутствуют в данных.")

        # Обработка пропущенных значений
        data = data.dropna(subset=['oiliness', 'moisture'])

        # Взятие нужных столбцов
        X = data[['oiliness', 'moisture']]

        # Применение PCA
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)

        # Визуализация данных
        plt.figure(figsize=(8, 6))
        plt.scatter(components[:, 0], components[:, 1], alpha=0.7)
        plt.title('PCA of Skin Data')
        plt.xlabel('PC1')
        plt.ylabel('PC2')

        # Сохранение графика
        os.makedirs('static', exist_ok=True)
        plot_path = 'static/pca_plot.png'
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    except FileNotFoundError:
        raise Exception("Файл данных не найден.")
    except Exception as e:
        raise Exception(f"Ошибка при визуализации данных: {e}")
