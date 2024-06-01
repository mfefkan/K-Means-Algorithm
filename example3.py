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

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        exampleDataSet (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in exampleDataSet. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n))

    # Compute the new centroids
    for k in range(K):   
        points = X[idx == k+1]  # Get a list of all data points in exampleDataSet assigned to centroid k  
        if len(points) > 0:  # Ensure there are points assigned to this centroid
            centroids[k] = np.mean(points, axis=0)  # Compute the mean of the points assigned

    return centroids

 
K = initial_centroids.shape[0]
centroids = compute_centroids(exampleDataSet, idx, K)  

print("Yeni centroids:")
print(centroids)

 
def plot_progress_kMeans(exampleDataSet, centroids, previous_centroids, idx, K, iteration):
    """
    A helper function that displays progress of K-Means as it is running. It is 
    intended for use only with 2D data. It plots data points with colors assigned 
    to each centroid. With each iteration of K-Means, the centroids are moved 
    and the assignment of points is updated accordingly.
    """

    # Plot the exampleDataSet
    plt.scatter(exampleDataSet[:, 0], exampleDataSet[:, 1], c=idx, cmap='viridis', marker='o', alpha=0.6)

    # Plot the centroids as black x's
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', s=100, linewidth=1.5)

    # Plot the history of the centroids with lines
    for i in range(centroids.shape[0]):
        plt.plot([centroids[i, 0], previous_centroids[i, 0]], [centroids[i, 1], previous_centroids[i, 1]], c='k', linestyle='--', linewidth=1)
 
    # Plot settings
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Iteration %d' % iteration)
    plt.grid(True)
    #plt.show()


def run_kMeans(exampleDataSet, initial_centroids, max_iters=5, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = exampleDataSet.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(exampleDataSet, centroids)
        
        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(exampleDataSet, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        
        for x in range(centroids.shape[0]):
            plt.text(centroids[x, 0], centroids[x, 1], f'{i}', fontsize=16, ha='left')
  
         
        # Given the memberships, compute new centroids
        centroids = compute_centroids(exampleDataSet, idx, K)
    plt.show() 
    return centroids, idx


# Number of iterations
max_iters = 5

# Run K-Means
centroids, idx = run_kMeans(exampleDataSet, initial_centroids, max_iters, plot_progress=True)