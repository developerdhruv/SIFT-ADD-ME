import cv2

def visualize_keypoints_and_matches(img1, kp1, img2, kp2, matches):
    img1_kp = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    img2_kp = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow('Keypoints - Group Image', img1_kp)
    cv2.imshow('Keypoints - User Image', img2_kp)
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

