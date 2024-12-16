 import numpy as np
 import matplotlib.pyplot as plt
 import cv2
 # Load and preprocess image
 image = cv2.imread('house.jpg', cv2.IMREAD_GRAYSCALE)
 if image is None:
     raise FileNotFoundError("The image file 'house.jpg' was not found.")
 _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
 # Structuring element and operations
 struct_elem = np.ones((3, 3), np.uint8)
 eroded_image = cv2.erode(binary_image, struct_elem)
 dilated_image = cv2.dilate(binary_image, struct_elem)
 # Differences
 eroded_diff = eroded_image - binary_image
 dilated_diff = dilated_image - binary_image
 # Display results
 titles = ['Original Binary', 'Eroded', 'Dilated', 'Erosion Diff', 'Dilation Diff']
 images = [binary_image, eroded_image, dilated_image, eroded_diff, dilated_diff]
 plt.figure(figsize=(16, 6))
 for i, img in enumerate(images):
   plt.subplot(2, 3, i + 1)
   plt.imshow(img, cmap='gray')
   plt.title(titles[i])
   plt.axis('off')
 plt.tight_layout()
 plt.show()
