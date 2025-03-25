import numpy as np
from skimage.metrics import structural_similarity
import cv2
import argparse

def calculate_ssim(img1_path, img2_path):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    
    """
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not read one or both of the images.")
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    if gray1.shape != gray2.shape:
        raise ValueError("Images must have the same dimensions.")
    
    # Calculate SSIM
    (score, diff) = structural_similarity(gray1, gray2, full=True)
    

    diff = (diff * 255).astype("uint8")
    
    print(f"SSIM: {score}")
    return score
