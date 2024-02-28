from matplotlib import pyplot as plt
import numpy as np
import cv2
from Task1_1 import convolution_2d
import time

# image = cv2.imread('victoria.png', cv2.IMREAD_COLOR)
image = cv2.imread('R.jpeg', cv2.IMREAD_COLOR)
# image = cv2.imread('victoria2.jpg', cv2.IMREAD_COLOR)
#
image = cv2.resize(image, (200, 200))
# image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define a kernel (e.g., a simple 3x3 averaging filter)
# kernel = np.array([[1, 0, -1],
#                    [0, 0, 0],
#                    [-1, 0, 1]])
# Perform convolution
# result_color = convolution_2d(image_color, kernel)
# result_2d = cv2.filter2D(image_color, -1, kernel)


sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
kernel = sharpen_kernel
start_time = time.time()
result_gray = convolution_2d(image_gray, kernel)
end_time = time.time()
print("Time taken for convolution_2d: ", end_time - start_time)

start_time = time.time()
result_gray_2d = cv2.filter2D(image_gray, -1, kernel)
end_time = time.time()
print("Time taken for cv2.filter : ", end_time - start_time)

# Display the images for comparison
# plt.subplot(131), plt.imshow(image_color), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.imshow(result_color), plt.title('Convolution')
# plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(result_2d), plt.title('cv2.filter2D')
# plt.xticks([]), plt.yticks([])
# plt.show()
#
plt.subplot(131), plt.imshow(image_gray, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(result_gray, cmap='gray'), plt.title('Convolution')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(result_gray_2d, cmap='gray'), plt.title('cv2.filter2D')
plt.xticks([]), plt.yticks([])
plt.show()

kernel = np.array([[1, 0, -1],
                   [0, 0, 0],
                   [-1, 0, 1]])

image = np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15],
                  [16, 17, 18, 19, 20],
                  [21, 22, 23, 24, 25]], dtype=np.uint8)
result = convolution_2d(image.astype(np.float32), kernel)
result_2d = cv2.filter2D(image, -1, kernel)
print(result_2d)
print(result)
