import cv2  
import os  
import numpy as np  

tiles = [cv2.imread(f"test_images/upscaled/tile_{i}.jpg") for i in range(16)]  # Adjust based on tile count  
stitched = cv2.vconcat([cv2.hconcat(tiles_row) for tiles_row in np.array_split(tiles, 4)])  # 4x4 grid  
cv2.imwrite("test_images/final_output.jpg", stitched)  