import cv2
from matplotlib import pyplot as plt

imagine1 = cv2.imread('victoria1.jpg', cv2.IMREAD_COLOR)
imagine1 = cv2.cvtColor(imagine1, cv2.COLOR_BGR2RGB)
imagine2 = cv2.imread('victoria2.jpg', cv2.IMREAD_COLOR)
imagine2 = cv2.cvtColor(imagine2, cv2.COLOR_BGR2RGB)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(imagine1, None)
kp2, des2 = orb.detectAndCompute(imagine2, None)

img1 = cv2.drawKeypoints(imagine1, kp1, None, flags=0)
img2 = cv2.drawKeypoints(imagine2, kp2, None, flags=0)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
ax1.set_title("Keypoints in Victoria1.jpg")
ax1.imshow(img1)
ax2.set_title("Keypoints in Victoria2.jpg")
ax2.imshow(img2)

plt.show()



