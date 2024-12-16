import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
# Load the image
image = imread('house.jpg')
# Convert the image to grayscale
gray = rgb2gray(image) * 255 
# Convert to grayscale and scale to 0-255
# Enhance contrast using min-max stretching
enhanced = ((gray - np.min(gray)) / (np.max(gray) - np.min(gray)) * 255).astype(np.uint8)
     # Segment using thresholding (simple mean threshold)
segmented = enhanced > np.mean(enhanced)
# Plot the images
plt.figure(figsize=(15, 5))
# Original Image
plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')
# Enhanced Image
plt.subplot(1, 3, 2)
plt.imshow(enhanced, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')
# Segmented Image
plt.subplot(1, 3, 3)
plt.imshow(segmented, cmap='gray')
plt.title('Segmented Image')
plt.axis('off')
plt.tight_layout()
plt.show()
