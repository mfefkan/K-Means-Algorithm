import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Örnek bir veri seti oluşturalım 
m = 100  # Veri setindeki örnek sayısı
n = 2  # Veri setindeki özellik sayısı
exampleDataSet = np.random.randint(0, 100, size=(m, n))  # 100x2 boyutunda rastgele bir veri seti oluşturalım


def find_closest_centroids(X, centroids):
    """
    Veri noktalarını en yakın merkezlere atar.

    Parametreler:
    - X: Özellik matrisi (örnekler satır, özellikler sütun)
    - centroids: Merkezler matrisi (K merkez, özellikler sütun)

    Çıktı:
    - idx: Her bir veri noktasının en yakın merkez indeksi (1'den K'ye)
    """ 
    K = centroids.shape[0]  # Merkezlerin sayısı
    idx = np.zeros(X.shape[0])  # Her bir veri noktasının atandığı merkezin indeksini tutan dizi

    for i in range(X.shape[0]):
        # Her bir veri noktası için en yakın merkezi bul
        distances = np.linalg.norm(X[i] - centroids, axis=1)  # Öklid uzaklığını hesapla
        idx[i] = np.argmin(distances) + 1  # En küçük uzaklık değerine sahip merkezin indeksini kaydet 
        
    return idx


# Example centroids
initial_centroids = np.array([[25,25], [50,50], [75,75]])
# find_closest_centroids calling
idx = find_closest_centroids(exampleDataSet, initial_centroids)

# Results
print("Her bir veri noktasinin en yakin merkez indeksi:")
print(idx)