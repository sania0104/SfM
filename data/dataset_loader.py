import cv2
import os
import re

def load_images(folder_path, count=None):
    images = []
    files = os.listdir(folder_path)
    
    # Natural Sort Key 
    def natural_key(string):
        # turns "frame10.jpg" into ['frame', 10, '.jpg'] so it sorts by number
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', string)]
    
    files = sorted(files, key=natural_key)
    
    if count:
        files = files[:count]
        
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, f)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"[WARN] Could not read image: {img_path}")
            else:
                images.append(img)
    
    return images
