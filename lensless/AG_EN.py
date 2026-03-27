import cv2
import numpy as np

###计算输入图像image的平均梯度
def compute_average_gradient(image):
    #转换图像为灰度 ， 视情况而定
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = image
    #print(image.shape)
    # 计算梯度
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度的幅值
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # 计算平均梯度
    average_gradient = np.mean(gradient_magnitude)

    return average_gradient

###计算输入图像image的平均梯度
def calculate_entropy(image):
    #转换图像为灰度 ， 视情况而定
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算图像的信息熵
    hist, _ = np.histogram(image.flatten(), bins=65536, range=[0,65535])
    hist = hist / float(np.sum(hist + 1e-8))  # 防止除零错误
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    return entropy


#调用示例
#I = cv2.imread('')
#AG_I = compute_average_gradient(I)
#EN_I = calculate_entropy(I)