import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dataset_path = "C:\dataset"

image_size = (100, 100)  # Ukuran gambar yang akan diubah

def load_dataset(dataset_path):
    images = []
    labels = []
    class_names = os.listdir(dataset_path)
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, image_size)
                    images.append(img_resized.flatten())
                    labels.append(label)
    return np.array(images), np.array(labels), class_names

print("Memuat dataset...")
X, y, class_names = load_dataset(dataset_path)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

print("Melatih model SVM...")
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

print("Memvalidasi model...")
y_pred = svm_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Akurasi Validasi: {accuracy * 100:.2f}%")

def predict_signature(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img_resized = cv2.resize(img, image_size).flatten()
        prediction = svm_model.predict([img_resized])
        return class_names[prediction[0]]
    return None

# Meminta input dari pengguna untuk path gambar
image_to_test = input("Masukkan path gambar tanda tangan untuk prediksi: ")
predicted_class = predict_signature(image_to_test)
if predicted_class is not None:
    print(f"Prediksi kelas tanda tangan: {predicted_class}")
else:
    print("Error: Tidak dapat memproses gambar. Silakan periksa path atau file gambar.")