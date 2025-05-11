from flask import Flask, render_template, request, redirect, url_for
import os
import sys

sys.path.append('.')

app = Flask(__name__)
UPLOAD_FOLDER = 'data/skin_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Импорт всех алгоритмов
from supervised_learning.linear_regression import predict_hydration
from supervised_learning.logistic_regression import classify_skin
from supervised_learning.decision_tree import recommend_product
from supervised_learning.random_forest import multi_classify_skin
from supervised_learning.naive_bayes import analyze_review
from supervised_learning.knn import find_similar_clients
from supervised_learning.svm import precise_classify_skin
from supervised_learning.gradient_boosting import predict_product_effectiveness
from unsupervised_learning.kmeans_clustering import segment_clients
from unsupervised_learning.apriori_rules import find_product_associations
from unsupervised_learning.pca_analysis import visualize_data
from computer_vision.sift_analysis import analyze_skin_texture


@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}

    if request.method == 'POST':
        # Linear Regression
        if 'predict_hydration' in request.form:
            oil = float(request.form['oil_lr'])
            results['hydration'] = predict_hydration(oil)

        # Logistic Regression
        elif 'classify_skin' in request.form:
            oil = float(request.form['oil_lr2'])
            moisture = float(request.form['moisture_lr2'])
            results['skin_type'] = classify_skin(oil, moisture)

        # Decision Tree
        elif 'recommend_product' in request.form:
            oiliness = float(request.form['oiliness_dt'])
            moisture = float(request.form['moisture_dt'])
            skin_type = int(request.form['skin_type_dt'])
            results['product_recommendation'] = recommend_product(oiliness, moisture, skin_type)



        # Random Forest
        elif 'multi_classify_skin' in request.form:
            oil = float(request.form['oil_rf'])
            moisture = float(request.form['moisture_rf'])
            results['multi_skin_type'] = multi_classify_skin(oil, moisture)

        # Naive Bayes
        elif 'analyze_review' in request.form:
            review = request.form['review']
            results['sentiment'] = analyze_review(review)

        # KNN
        elif 'find_similar_clients' in request.form:
            oil = float(request.form['oil_knn'])
            moisture = float(request.form['moisture_knn'])
            results['similar_clients'] = find_similar_clients(oil, moisture)

        # SVM
        elif 'precise_classify_skin' in request.form:
            oil = float(request.form['oil_svm'])
            moisture = float(request.form['moisture_svm'])
            results['precise_skin_type'] = precise_classify_skin(oil, moisture)

        # Gradient Boosting
        elif 'predict_product_effectiveness' in request.form:
            oil = float(request.form['oil_gb'])
            moisture = float(request.form['moisture_gb'])
            results['effectiveness'] = predict_product_effectiveness(oil, moisture)

        # K-means
        elif 'segment_clients' in request.form:
            results['client_segments'] = segment_clients()

        # Apriori
        elif 'find_product_associations' in request.form:
            results['associations'] = find_product_associations()

        # PCA
        elif 'visualize_data' in request.form:
            results['pca_plot'] = visualize_data()

        # SIFT
        elif 'upload_image' in request.form:
            if 'image' in request.files:
                file = request.files['image']
                if file.filename != '':
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(image_path)
                    results['sift_analysis'] = analyze_skin_texture(image_path)

    return render_template('index.html', **results)


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)