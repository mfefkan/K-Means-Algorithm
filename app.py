import numpy as np
import matplotlib.pyplot as plt
from utils import *
 

def find_closest_centroids(X, centroids):
    """
    Assigns data points to the closest centroids.

    Parameters:
    - X: Feature matrix (examples as rows, features as columns)
    - centroids: Matrix of centroids (K centroids, features as columns)

    Output:
    - idx: Index of the closest centroid for each data point (from 0 to K-1)
    """ 
    K = centroids.shape[0]  # Number of centroids
    idx = np.zeros(X.shape[0], dtype=int)  # Array to hold the index of the centroid each data point is assigned to

    for i in range(X.shape[0]):
        # Find the closest centroid for each data point
        distances = np.linalg.norm(X[i] - centroids, axis=1)  # Compute the Euclidean distance
        idx[i] = np.argmin(distances)   # Store the index of the centroid with the smallest distance
        
    return idx

  
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
        points = X[idx == k]  # Get a list of all data points in exampleDataSet assigned to centroid k  
        if len(points) > 0:  # Ensure there are points assigned to this centroid
            centroids[k] = np.mean(points, axis=0)  # Compute the mean of the points assigned

    return centroids

  
 
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
    return centroids, idx
 
def kMeans_init_centroids(X, K):
    # X'den rastgele K adet veri noktası seç
    m = X.shape[0]
    indices = np.random.choice(m, K, replace=False)
    centroids = X[indices]
    return centroids


def plot_kMeans_RGB(X_img, centroids, idx, K):
    img_recreated = centroids[idx.astype(int)].reshape((original_img.shape[0], original_img.shape[1], 3))
    plt.imshow(img_recreated)
    plt.title(f'Recreated Image with {K} colors')
    plt.show()


def show_centroid_colors(centroids):
    # Her bir centroid'i bir kare olarak çiz
    n = centroids.shape[0]
    fig, ax = plt.subplots(1, n, figsize=(15, 2))
    for i in range(n):
        ax[i].imshow(np.ones((10, 10, 3), dtype=np.float32) * centroids[i])
        ax[i].axis('off')
    plt.show()

# Load an image of a bird
original_img = plt.imread('C:/Users/MFE/Downloads/bird_small.png')

# Visualizing the image
plt.imshow(original_img)

print("Shape of original_img is:", original_img.shape)

# Divide by 255 so that all values are in the range 0 - 1 (not needed for PNG files)
# original_img = original_img / 255

# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.
X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

# Using the function you have implemented above. 
initial_centroids = kMeans_init_centroids(X_img, K)

# Run K-Means - this can take a couple of minutes depending on K and max_iters
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)

print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])

# Plot the colors of the image and mark the centroids
plot_kMeans_RGB(X_img, centroids, idx, K)

# Visualize the 16 colors selected
show_centroid_colors(centroids)

# Find the closest centroid of each pixel
idx = find_closest_centroids(X_img, centroids)

# Ensure idx is integer type
idx = idx.astype(int)

# Replace each pixel with the color of the closest centroid
X_recovered = centroids[idx, :] 


# Reshape image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape) 

# Display original image
fig, ax = plt.subplots(1,2, figsize=(16,16))
#plt.axis('off') 

ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].axis('off')


# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].axis('off')
