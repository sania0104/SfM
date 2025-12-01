import cv2

def detect_and_compute(image, max_features=40000):
    detector = cv2.SIFT_create(max_features, contrastThreshold=0.006, edgeThreshold=25, sigma=1.6)

    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors