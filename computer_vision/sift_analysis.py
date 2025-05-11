import cv2
import os

def analyze_skin_texture(image_path):
    """Анализирует текстуру кожи с помощью SIFT."""
    if not os.path.exists(image_path):
        raise Exception(f"Изображение {image_path} не найдено.")

    # Чтение изображения
    img = cv2.imread(image_path)

    # Преобразование в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Инициализация SIFT
    sift = cv2.SIFT_create()

    # Выявление ключевых точек и дескрипторов
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Возвращаем количество найденных ключевых точек
    return len(keypoints)

def visualize_keypoints(image_path, output_image_path):
    """Визуализирует ключевые точки на изображении и сохраняет результат."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)

    # Наложение ключевых точек на изображение
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)

    # Сохраняем изображение с ключевыми точками
    cv2.imwrite(output_image_path, img_with_keypoints)
    print(f"Изображение с ключевыми точками сохранено как {output_image_path}")

# Пример использования:
image_paths = [
    'data/dry_skin.jpg',
    'data/oily_skin.jpg',
    'data/normal_skin.jpg'
]

# Создаем или открываем файл для записи результатов
with open('keypoints_results.txt', 'w') as results_file:
    for image_path in image_paths:
        keypoint_count = analyze_skin_texture(image_path)
        results_file.write(f"{image_path}: {keypoint_count} ключевых точек\n")
        print(f"{image_path}: {keypoint_count} ключевых точек")

        # Визуализируем ключевые точки и сохраняем изображение
        output_image_path = f"output/{os.path.basename(image_path)}_keypoints.jpg"
        os.makedirs('output', exist_ok=True)  # Создаем папку для сохраненных изображений
        visualize_keypoints(image_path, output_image_path)
