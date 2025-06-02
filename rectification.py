import cv2
import numpy as np
import os

os.makedirs("rectified", exist_ok=True)

def extract_feature_points(img1, img2):

################ Extract feature points using SIFT and match them using FLANN. #################

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    pts1 = []
    pts2 = []

    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)
            pts1.append(keypoints1[m.queryIdx].pt)
            pts2.append(keypoints2[m.trainIdx].pt)

    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)


    return pts1, pts2, keypoints1, keypoints2, good_matches

def compute_fundamental_matrix(pts1, pts2):

################# Compute the fundamental matrix using RANSAC.#################

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)

    if F is None or F.shape != (3, 3):
        raise ValueError("Estimation de la matrice fondamentale échouée.")
    return F

def stereorectification(img1, img2, pts1, pts2, F):

#################  Rectifies stereo images using cv2.stereoRectifyUncalibrated(). #################
#################  Applies normalization to avoid excessive distortion.           #################

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    success, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1)
    )

    if not success:
        raise ValueError("La rectification a échoué.")

    H1 /= H1[2, 2]
    H2 /= H2[2, 2]

    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1), flags=cv2.INTER_LINEAR)
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2), flags=cv2.INTER_LINEAR)

    cv2.imwrite("rectified/rectified_left.png", img1_rectified)
    cv2.imwrite("rectified/rectified_right.png", img2_rectified)

    return img1_rectified, img2_rectified, H1, H2
