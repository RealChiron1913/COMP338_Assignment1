import cv2
import time
import numpy as np
from matplotlib import pyplot as plt


# Function to load images, extract keypoints and descriptors using SIFT and ORB
def extract_features(image_path, method="SIFT"):
    # Load the image
    image = cv2.imread(image_path)

    # Convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the detector
    if method == "SIFT":
        detector = cv2.SIFT_create()
    elif method == "ORB":
        detector = cv2.ORB_create()

    # Detect keypoints and extract descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    return keypoints, descriptors


# Function to perform keypoint matching using Brute-Force Matcher
def match_keypoints(desc1, desc2, method="SIFT"):
    # Create BFMatcher object
    if method == "SIFT":
        # Since SIFT uses L2 norm
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif method == "ORB":
        # ORB uses Hamming norm
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(desc1, desc2)

    # Sort them in the order of their distance (best matches first).
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


# Paths to the image files
image_path1 = 'victoria1.jpg'
image_path2 = 'victoria2.jpg'

# Extract features using SIFT
start_time = time.time()
keypoints1_sift, descriptors1_sift = extract_features(image_path1, "SIFT")
keypoints2_sift, descriptors2_sift = extract_features(image_path2, "SIFT")
end_time = time.time()
print("Time taken to extract SIFT features: ", end_time - start_time)


# Extract features using ORB
start_time = time.time()
keypoints1_orb, descriptors1_orb = extract_features(image_path1, "ORB")
keypoints2_orb, descriptors2_orb = extract_features(image_path2, "ORB")
end_time = time.time()
print("Time taken to extract ORB features: ", end_time - start_time)

# Match keypoints using SIFT descriptors
matches_sift = match_keypoints(descriptors1_sift, descriptors2_sift, "SIFT")
print('SIFT',len(matches_sift))

# Match keypoints using ORB descriptors
matches_orb = match_keypoints(descriptors1_orb, descriptors2_orb, "ORB")
print('ORB',len(matches_orb))

image1 = cv2.imread(image_path1, cv2.IMREAD_COLOR)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

img_matches_sift = cv2.drawMatches(image1, keypoints1_sift, image2, keypoints2_sift, matches_sift[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches_orb = cv2.drawMatches(image1, keypoints1_orb, image2, keypoints2_orb, matches_orb[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_matches_sift), plt.title('Matches with SIFT')
plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(img_matches_orb), plt.title('Matches with ORB')
plt.xticks([]), plt.yticks([])
plt.show()

