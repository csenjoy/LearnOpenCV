import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def GenerateBlack8x8Numpy():
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

def Convert2LenaGray():
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

def GenerateBlueColorImage():
    # 生成blue彩色图300x300 3个颜色通道BGR
    # opencv中颜色通道顺序BGR
    blue = np.zeros((300,300,3), dtype=np.uint8)
    blue[:,:,0]=255#设置红色blue[:,:2] = 255

    print("blue=",blue)
    #显示红色而不是蓝色，因为pyplot以RGB方式展示
    plt.imshow(blue)
    plt.show()

    #使用cv2显示
    cv2.imshow("blueInCv2", blue)
    #没有waitKey时，看不到blue显示，需要设置展示2000ms
    cv2.waitKey(500)

    #转为RGB显示
    blue = cv2.cvtColor(blue, cv2.COLOR_BGR2RGB)
    plt.imshow(blue)
    plt.show()

def GenerateLenaFaceROI():
    #提取感兴趣区域ROI
    lena = cv2.imread("./lena.jpg")

    lenaShowInPlt = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
    plt.imshow(lenaShowInPlt)
    plt.show()
    face = lena[100:180, 100:180]

    faceShowInPlt = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    plt.imshow(faceShowInPlt)
    plt.show()
    cv2.imwrite("./lena_face.jpg", face)

    lena[100:180, 100:180] = np.random.randint(0, 256, (80, 80, 3))
    lenaShowInPlt = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
    plt.imshow(lenaShowInPlt)
    plt.show()
    cv2.imwrite("./lena_mosaic.jpg", lena)
    
def SplitLenaColorChannels():
    #通道分离
    lena = cv2.imread("./lena.jpg", -1)
    b = lena[:,:,0]
    g = lena[:,:,1]
    r = lena[:,:,2]
    #使用函数通道拆分
    b,g,r = cv2.split(lena)
    plt.gray()
    plt.subplot(1,3,1)
    plt.imshow(b)
    plt.subplot(1,3,2)
    plt.imshow(g)
    plt.subplot(1,3,3)
    plt.imshow(r)
    plt.show()
    
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
    lena[:,:,0] = 0
    plt.subplot(1,3,1)
    plt.imshow(lena)
    lena[:,:,1] = 0
    plt.subplot(1,3,2)
    plt.imshow(lena)
    lena[:,:,2] = 0
    plt.subplot(1,3,3)
    plt.imshow(lena)
    plt.show()

def MergeLenaColorChannles():
    lena = cv2.imread("./lena.jpg")
    b,g,r = cv2.split(lena)
    bgr = cv2.merge([b,g,r])
    rgb = cv2.merge([r,g,b])
    plt.subplot(1,2,1)
    plt.imshow(bgr)
    plt.subplot(1,2,2)
    plt.imshow(rgb)
    plt.show()

def GetImageAttributeOfLena():
    #获取灰度图像的属性
    grayLena = cv2.imread("./lena.jpg", 0)
    colorLena = cv2.imread("./lena.jpg", -1)
    print("GrayLena Attributes: ")
    print("grayLena.shape={}", format(grayLena.shape))
    print("grayLena.size={}",format(grayLena.size))
    print("grayLena.dtype={}", format(grayLena.dtype))
    plt.gray()
    plt.imshow(grayLena)
    plt.show()
    
    #获取彩色图像的属性
    colorLena = cv2.cvtColor(colorLena, cv2.COLOR_BGR2RGB)
    print("colorLena.shape={}", format(colorLena.shape))
    print("colorLena.size={}",format(colorLena.size))
    print("colorLena.dtype={}", format(colorLena.dtype))
    plt.imshow(colorLena)
    plt.show()

#SplitLenaColorChannels()
#MergeLenaColorChannles()
GetImageAttributeOfLena()