import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Data contoh (dapat disesuaikan dengan data penipuan marketplace yang lebih relevan)
data = {
    'jumlah_transaksi': [100, 120, 130, 90, 5000, 100, 110, 95, 105, 150, 200, 100, 95, 3000],
    'frekuensi_transaksi': [1, 1, 2, 1, 10, 1, 1, 1, 2, 1, 1, 1, 1, 20],
    'waktu_transaksi': [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    'status': ['Normal', 'Normal', 'Normal', 'Normal', 'Penipuan', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Penipuan']
}

df = pd.DataFrame(data)

print("Data Transaksi:")
print(df)

# Fitur dan target
X = df[['jumlah_transaksi', 'frekuensi_transaksi', 'waktu_transaksi']]
y = df['status'].map({'Normal': 0, 'Penipuan': 1})  # 0 = Normal, 1 = Penipuan

# Pembagian data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standarisasi data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Membangun model Neural Network (MLP)
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Prediksi
predictions = model.predict(X_test_scaled)

# Hasil deteksi
df['prediksi'] = model.predict(scaler.transform(X[['jumlah_transaksi', 'frekuensi_transaksi', 'waktu_transaksi']]))
df['status_prediksi'] = df['prediksi'].apply(lambda x: 'Normal' if x == 0 else 'Penipuan')

print("\nHasil Prediksi untuk Seluruh Data:")
print(df)

# Evaluasi model
print("\nEvaluasi Model:")
print(classification_report(y_test, predictions))

def deteksi_penipuan(jumlah_transaksi, frekuensi_transaksi, waktu_transaksi):
    input_data = np.array([[jumlah_transaksi, frekuensi_transaksi, waktu_transaksi]])
    input_scaled = scaler.transform(input_data)
    
    prediksi = model.predict(input_scaled)
    
    if prediksi == 0:
        return "Transaksi Normal"
    else:
        return "Transaksi Mencurigakan (Penipuan)"

print("\nMasukkan data transaksi untuk deteksi penipuan:")
jumlah_transaksi = float(input("Jumlah Transaksi: "))
frekuensi_transaksi = int(input("Frekuensi Transaksi: "))
waktu_transaksi = int(input("Waktu Transaksi (1 untuk normal, 2 untuk anomali): "))

hasil = deteksi_penipuan(jumlah_transaksi, frekuensi_transaksi, waktu_transaksi)
print(f"Hasil Prediksi: {hasil}")

# Tema : Deteksi Penipuan Online Marketplace Menggunakan Algoritma Neural Network
# Penipuan di marketplace online merupakan salah satu masalah yang semakin berkembang 
# seiring dengan meningkatnya transaksi digital. Di pasar yang besar ini, banyak pelaku 
# penipuan yang mencoba mengeksploitasi celah-celah dalam sistem untuk mendapatkan keuntungan 
# secara tidak sah. Oleh karena itu, deteksi dini terhadap transaksi yang mencurigakan sangat 
# penting untuk melindungi konsumen, pedagang, dan platform marketplace itu sendiri.