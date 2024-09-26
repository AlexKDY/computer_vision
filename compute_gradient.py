import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_derivative(img, filter):
    if len(img.shape) == 3:
        height, width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        height, width = img.shape 
    u, v = filter.shape
    y = height - v + 1
    x = width - u + 1
    result = np.zeros((y, x))
    for j in range(y):
        for i in range(x):
            result[j, i] = np.sum(img[j : j + u, i : i + v] * filter)
    
    return result
               
def compute_gradients(img):
    vertical_derivative = np.array([
                            [-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]
                          ])
    horiziontal_derivative = np.array([
                            [-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]
                          ])
        
    I_y = compute_derivative(img, vertical_derivative)
    I_x = compute_derivative(img, horiziontal_derivative)
    magnitude = np.sqrt(I_y ** 2 + I_x ** 2)
    direction = np.arctan2(I_y, I_x)
    
    return magnitude, direction

if __name__ == '__main__':
    input_image = cv2.imread('image/lenna.png')  # BGR 이미지로 읽기
    magnitude, direction = compute_gradients(input_image)
    
    plt.figure(figsize=(16, 4))
    plt.subplot(141), plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(142), plt.imshow(magnitude, cmap='gray'), plt.title('Gradient Magnitude')
    plt.subplot(143), plt.imshow(direction, cmap='hsv'), plt.title('Gradient Direction')
    plt.show()