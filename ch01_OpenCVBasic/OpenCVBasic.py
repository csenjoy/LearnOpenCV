import cv2

import matplotlib.pyplot as plt
# read image from filesystem
# flags -1, 以原格式读取example.png
lena = cv2.imread("./example.png", -1)
#print(lena)

# show image
if lena is not None:
    cv2.namedWindow("example")
    cv2.imshow("example", lena)
    # show window delay 2000ms
    cv2.waitKey(2000)

    # save to filesystem
    cv2.imwrite("./image_save.png", lena)
    
    # show lena using pyplot
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
    plt.imshow(lena)
    plt.show()