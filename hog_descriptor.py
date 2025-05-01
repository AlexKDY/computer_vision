import cv2
import numpy as np
import matplotlib.pyplot as plt
from compute_gradient import compute_gradients

bins = [angle for angle in range(0, 180, 20)]

def calculate_hog(img, block_size, cell_size):
    if len(img.shape) == 3:
        height, width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        height, width = img.shape
    
    result = np.zeros(())
    
    for j in range(0, height, cell_size):
        for i in range(0, width, cell_size):
            magnitude, direction = compute_gradients(img[j : j + cell_size, i : i + cell_size])
            
    
if __name__ == '__main__':
    input_image = cv2.imread('image/lenna.png')