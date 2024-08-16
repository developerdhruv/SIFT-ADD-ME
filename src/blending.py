import cv2
import numpy as np

def blend_images(group_image_path, warped_user_image):
    group_image = cv2.imread(group_image_path)

   
    gray_user_image = cv2.cvtColor(warped_user_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_user_image, 1, 255, cv2.THRESH_BINARY)

    
    blended_image = cv2.seamlessClone(warped_user_image, group_image, mask, (group_image.shape[1] // 2, group_image.shape[0] // 2), cv2.NORMAL_CLONE)

    return blended_image
