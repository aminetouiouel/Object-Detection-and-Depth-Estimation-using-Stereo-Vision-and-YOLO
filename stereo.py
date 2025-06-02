import cv2
import numpy as np

def compute_disparity_map(left_img, right_img):

################# compute the disparity map using StereoSGBM #################

    if len(left_img.shape) == 3:
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    if len(right_img.shape) == 3:
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(numDisparities=16*5,blockSize=9)


    disparity = stereo.compute(left_img, right_img).astype(np.float32)

    valid_disp = disparity[disparity > 0]
    if valid_disp.size == 0:
        raise ValueError("No valid disparities found.")

    norm_image = cv2.normalize(disparity, None, 4, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return norm_image


def cost_aggregation(disparity_map, window_size=5):

    aggregated_map = cv2.boxFilter(disparity_map, -1, (window_size, window_size), normalize=True)
    return aggregated_map

def compute_depth_map(disparity_map, focal_length, baseline):

    epsilon = 1e-6
    depth_map = (focal_length * baseline) / (disparity_map + epsilon)
    return depth_map
