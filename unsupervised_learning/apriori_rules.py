import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import MultiLabelBinarizer


def find_product_associations():
    """
    Находит ассоциации между продуктами на основе транзакций

    Returns:
        list: Список ассоциативных правил с метриками
    """
    try:
        # Загрузка данных
        data = pd.read_csv('data/transactions.csv')

        # Преобразование строки продуктов в списки
        data['products'] = data['products'].str.split(', ')

        # Создание бинарной матрицы с MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        product_matrix = mlb.fit_transform(data['products'])

        # Создание DataFrame с названиями продуктов
        product_df = pd.DataFrame(product_matrix, columns=mlb.classes_)

        # Поиск частых элементов
        frequent_itemsets = apriori(product_df, min_support=0.1, use_colnames=True)

        # Генерация правил ассоциаций
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

        # Форматирование результатов
        formatted_rules = []
        for _, rule in rules.iterrows():
            formatted_rules.append({
                'antecedents': list(rule['antecedents']),
                'consequents': list(rule['consequents']),
                'support': round(float(rule['support']), 3),
                'confidence': round(float(rule['confidence']), 3),
                'lift': round(float(rule['lift']), 3)
            })

        return formatted_rules

    except FileNotFoundError:
        raise Exception("Файл транзакций не найден. Проверьте путь к файлу 'data/transactions.csv'")
    except Exception as e:
        raise Exception(f"Ошибка анализа правил ассоциаций: {str(e)}")