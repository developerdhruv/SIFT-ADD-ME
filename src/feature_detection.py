import cv2

def detect_and_match_features(group_image_path, user_image_path):
    # Load images
    group_image = cv2.imread(group_image_path, cv2.IMREAD_GRAYSCALE)
    user_image = cv2.imread(user_image_path, cv2.IMREAD_GRAYSCALE)

    
    sift = cv2.SIFT_create()

    
    kp_group, des_group = sift.detectAndCompute(group_image, None)
    kp_user, des_user = sift.detectAndCompute(user_image, None)

   
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
   
    matches = bf.match(des_group, des_user)

   
    matches = sorted(matches, key=lambda x: x.distance)

    return kp_group, kp_user, matches
