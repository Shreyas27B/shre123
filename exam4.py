import cv2, numpy as np, matplotlib.pyplot as plt
 # Load grayscale image
 image = cv2.imread('daredevil.jpg', cv2.IMREAD_GRAYSCALE)
 # Define filters
 low_pass_kernel = np.ones((15, 15), dtype=np.float32) / 225  
 # Larger kernel for smoother low-pass
 high_pass_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
 # Apply filters
 low_pass = cv2.filter2D(image, -1, low_pass_kernel)
 high_pass = cv2.filter2D(image, -1, high_pass_kernel)
# Combine filters
band_pass = cv2.addWeighted(low_pass, 0.5, high_pass, 0.5, 0)
# Plot images
plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(1, 4, 2), plt.imshow(low_pass, cmap='gray'), plt.title('Low-Pass')
plt.subplot(1, 4, 3), plt.imshow(high_pass, cmap='gray'), plt.title('High-Pass')
plt.subplot(1, 4, 4), plt.imshow(band_pass, cmap='gray'), plt.title('Band-Pass')
plt.show()
