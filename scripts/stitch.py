import cv2
import os
import numpy as np

def stitch_tiles(input_dir, output_path, tile_count=16):
    tiles = []
    for i in range(tile_count):
        img = cv2.imread(os.path.join(input_dir, f"tile_{i}.jpg"))
        if img is None:
            raise ValueError(f"Could not load tile_{i}.jpg")
        tiles.append(img)
    tiles = np.array(tiles)
    grid_size = int(np.sqrt(tile_count))
    stitched = cv2.vconcat([cv2.hconcat(list(tiles_row)) for tiles_row in np.array_split(tiles, grid_size)])
    success = cv2.imwrite(output_path, stitched)
    if success:
        print("Stitch completed successfully")
    else:
        print("Failed to write output image")

if __name__ == "__main__":
    stitch_tiles("test_images/upscaled", "test_images/final_output.jpg")