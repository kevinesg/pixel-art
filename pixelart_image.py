from functions import pixel_art
import numpy as np
import cv2

NUM_CLUSTERS = 10   # Number of colors in the pixel art
SCALE = 4   # One square has a width and height both equal to SCALE (SCALE = 8 means 8x8 pixels)
#IMG_PATH = '''IMAGE FILE PATH'''
IMG_PATH = '''IMAGE FILE PATH'''


# Generate the pixel art
canvas = pixel_art(IMG_PATH, num_clusters=NUM_CLUSTERS, scale=SCALE)

# This part is optional; this displays the original image and pixel art side by side
img = cv2.imread(IMG_PATH)
combined_imgs = np.hstack([img / 255, canvas])
cv2.imshow('Original Image and Pixel Art comparison', combined_imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Alternatively, we can display only the generated pixel art
cv2.imshow('Pixel Art', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()