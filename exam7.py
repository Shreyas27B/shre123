 import cv2
 import numpy as np
 import matplotlib.pyplot as plt
 def transform_image(image, tx, ty, angle, scale):
    h, w = image.shape[:2]
    translate = cv2.warpAffine(image, np.float32([[1, 0, tx], [0, 1, ty]]), (w, h), borderValue=(255, 255, 255))
    rotate = cv2.warpAffine(image, cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale), (w, h), borderValue=(255, 255, 255))
    scaled = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return translate, rotate, scaled
 image = cv2.imread('house.jpg')
 if image is not None:
    tx, ty, angle, scale = 15, 30, 45, 1.2
    translated, rotated, scaled = transform_image(image, tx, ty, angle, scale)
    images = [image, translated, rotated, scaled]
    titles = ["Original", f"Translated (tx={tx}, ty={ty})", f"Rotated (angle={angle}°)", f"Scaled (scale={scale})"]
    plt.figure(figsize=(12, 12))
for i, img in enumerate(images):
   plt.subplot(2, 2, i + 1)
   plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
   plt.title(titles[i])
   plt.axis('off')
 plt.tight_layout()
 plt.show()
