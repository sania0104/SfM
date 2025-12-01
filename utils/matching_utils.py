import cv2

def match_features(des1, des2, method="BF", ratio_test=True, ratio=0.8):
    if method == "BF":
        norm = cv2.NORM_L2 if des1.dtype== 'float32' else cv2.NORM_HAMMING
        matcher = cv2.BFMatcher(norm, crossCheck=not ratio_test)
        matches = matcher.knnMatch(des1, des2, k=2) if ratio_test else matcher.match(des1, des2)
        
        if ratio_test:
            # Lowe's ratio test
            good_matches = []
            for m,n in matches:
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
            matches = good_matches
    else:
        raise ValueError("Unknown matching method")
    
    matches = sorted(matches, key=lambda x: x.distance)
    return matches