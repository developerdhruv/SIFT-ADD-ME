import cv2
import numpy as np

def align_images(group_image_path, user_image_path, kp_group, kp_user, matches):
    group_image = cv2.imread(group_image_path)
    user_image = cv2.imread(user_image_path)

   
    src_pts = np.float32([kp_user[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_group[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

   
    height, width, channels = group_image.shape
    warped_user_image = cv2.warpPerspective(user_image, H, (width, height))

    return warped_user_image
