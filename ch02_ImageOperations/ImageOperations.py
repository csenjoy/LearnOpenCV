import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#Generats a 8x8 numpy array to demonstrate image operations
img = np.zeros((8,8), dtype=np.uint8)
print("img=\n",img)
#set the colormap to 'gray'
plt.gray()
plt.imshow(img)
plt.show()

# change img[0,3] pixel to white color
img[0,3] = 255
plt.gray()
plt.imshow(img)
plt.show()

lena = np.array(Image.open("./lena.jpg").convert('L'))
print(lena)
plt.imshow(lena)
plt.show()
cv2.imwrite("./lena_gray.jpg", lena)

#将部分区域设置为白色
for row in range(40, 80):
    for column in range(40, 80):
        lena[row, column] = 255
plt.imshow(lena)
plt.show()
