import cv2
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread('joeee.jpg', 0)

# Apply Canny Edge Detection
edges = cv2.Canny(image, 100, 200)

# Apply Laplacian for Texture Detection
texture = cv2.Laplacian(image, cv2.CV_64F)

# Display results
titles = ['Original Image', 'Edges', 'Texture']
images = [image, edges, abs(texture)]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')

plt.show()
