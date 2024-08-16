import cv2
from src.feature_detection import detect_and_match_features
from src.image_alignment import align_images
from src.blending import blend_images
from src.ui import upload_images, display_result

def main():
    # Upload images
    group_photo_path, user_photo_path = upload_images()

    # Detect and match features
    kp_group, kp_user, matches = detect_and_match_features(group_photo_path, user_photo_path)

    # Align images
    warped_user_image = align_images(group_photo_path, user_photo_path, kp_group, kp_user, matches)

    # Blend images
    result_image = blend_images(group_photo_path, warped_user_image)

    # Save and display result
    result_image_path = 'result.jpg'
    cv2.imwrite(result_image_path, result_image)
    display_result(result_image_path)


def process_photos(group_photo_path, user_photo_path, output_path):
    # Detect and match features
    kp_group, kp_user, matches = detect_and_match_features(group_photo_path, user_photo_path)

    # Align images
    warped_user_image = align_images(group_photo_path, user_photo_path, kp_group, kp_user, matches)

    # Blend images
    result_image = blend_images(group_photo_path, warped_user_image)

    # Save result
    cv2.imwrite(output_path, result_image)
    print(f"Processed image saved to {output_path}")


group_photo_path = './data/grp.jpg'
user_photo_path = './data/ind.jpeg'
output_path = 'path_to_output_image.jpg'
process_photos(group_photo_path, user_photo_path, output_path)
