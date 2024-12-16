import cv2
 import numpy as np
 import matplotlib.pyplot as plt
 img=cv2.imread('ip2.jpg')
 plt.imshow(img)
 cv2.waitKey(0)
 newimg=np.asarray(img)
 X,Y,D=newimg.shape
 div1=X//2
 div2=Y//2
 top_left=newimg[:div1,:div2]
 top_right=newimg[:div1,div2:]
 bottom_left=newimg[div1:,:div2]
 bottom_right=newimg[div1:,div2:]
 div_img=[top_left,top_right,bottom_left,bottom_right]
 fig, axs = plt.subplots(2, 2, figsize=(10, 10))
 for idx, ax in enumerate(axs.flat):
 ax.imshow(div_img[idx])
 ax.set_title(f'Quadrant {idx+1}: {div_img[idx].shape}')
 ax.axis('on')
 plt.show()
