import os
import cv2
import numpy as np
from yolo import detect_objects, draw_detections
from sterio import compute_disparity_map, compute_depth_map
from rectification import extract_feature_points, compute_fundamental_matrix, stereorectification


########################### RECTIFICATION ###########################


def main():
    left_image_path = 'ACQUISITION/image_L/L10.png'
    right_image_path = 'ACQUISITION/image_R/R10.png'

    img1 = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Erreur de chargement des images.")
        return None, None

    pts1, pts2, _, _, _ = extract_feature_points(img1, img2)
    F = compute_fundamental_matrix(pts1, pts2)
    rect_left, rect_right, _, _ = stereorectification(img1, img2, pts1, pts2, F)

    return rect_left, rect_right


########################### MAIN ###########################

rectified_left, rectified_right = main()
if rectified_left is None or rectified_right is None:
    raise ValueError("Erreur dans la rectification.")


rectified_left_bgr = cv2.cvtColor(rectified_left, cv2.COLOR_GRAY2BGR)
rectified_right_bgr = cv2.cvtColor(rectified_right, cv2.COLOR_GRAY2BGR)

########################### YOLO detections ###########################

detections_left = detect_objects(rectified_left_bgr)
detections_right = detect_objects(rectified_right_bgr)

left_drawn = draw_detections(rectified_left.copy(), detections_left)
right_drawn = draw_detections(rectified_right.copy(), detections_right)


########################### DISPARITY & DEPTH ###########################

disparity_map = compute_disparity_map(rectified_left, rectified_right)
if disparity_map is None or disparity_map.size == 0:
    raise ValueError("Carte de disparité vide.")


focal_length = 455.128894  # pixels
baseline = 0.1  # mètres
epsilon = 1e-6

depth_map = compute_depth_map(disparity_map, focal_length, baseline)

###########################  Distance Estimation ###########################


original_left = cv2.imread('ACQUISITION/image_L/L10.png')
if original_left is None:
    raise ValueError("Erreur lors du chargement de l'image gauche originale.")
original_left_disp = original_left.copy()

for detection in detections_right:
    x1, y1, x2, y2, conf, cls_id = detection
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    win = 5
    h, w = disparity_map.shape
    x_start = max(center_x - win, 0)
    x_end = min(center_x + win, w)
    y_start = max(center_y - win, 0)
    y_end = min(center_y + win, h)

    patch = disparity_map[y_start:y_end, x_start:x_end]
    valid_disp = patch[patch > 1.0]

    if valid_disp.size > 0:
        disparity = np.median(valid_disp)
        distance = (focal_length * baseline) / (disparity + epsilon)
        print(f" Distance estimée : {distance:.2f} m")
        cv2.rectangle(original_left_disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(original_left_disp, f"{distance:.2f} m", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 0), 2)
    else:
        print(f"Aucune disparité valide pour l'objet à ({center_x}, {center_y})")


########################### Display  ###########################

try:
    # Normalization for Display
    norm_disp = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.namedWindow('Original Left with Distance', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Rectified Left', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Rectified Right', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Disparity Map', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)

    #  Display
    cv2.imshow('Original Left with Distance', original_left_disp)
    cv2.imshow('Rectified Left', rectified_left)
    cv2.imshow('Rectified Right', rectified_right)
    cv2.imshow('Disparity Map', norm_disp)
    cv2.imshow('Depth Map', norm_depth)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Erreur d'affichage : {e}")
