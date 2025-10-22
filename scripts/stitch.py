import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

print("Starting stitch")
sys.stdout.flush()


def stitch_tiles(input_dir: str, output_path: str) -> None:
    tiles: list[np.ndarray[Any, np.dtype[Any]]] = []
    tile_files = sorted(
        [f for f in Path(input_dir).iterdir() if f.name.startswith("tile_") and f.name.endswith(".jpg")],
        key=lambda f: int(f.name.partition("_")[2].partition(".")[0]),
    )
    if not tile_files:
        raise ValueError("No tile files found in input directory")
    for tile_file in tile_files:
        img = cv2.imread(str(tile_file))
        if img is None:
            raise ValueError(f"Could not load {tile_file}")
        tiles.append(img)
    tile_count = len(tiles)
    tiles_array = np.array(tiles)
    grid_size = int(np.sqrt(tile_count))
    if grid_size * grid_size != tile_count:
        raise ValueError(f"Tile count {tile_count} is not a perfect square")
    stitched = cv2.vconcat([cv2.hconcat(list(tiles_row)) for tiles_row in np.array_split(tiles_array, grid_size)])
    print("Stitched shape:", stitched.shape)
    print("About to write to", output_path)
    sys.stdout.flush()
    success = cv2.imwrite(output_path, stitched)
    if success:
        print("Stitch completed successfully")
    else:
        print("Failed to write output image")
    sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 stitch.py <input_dir> <output_path>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_path = sys.argv[2]
    stitch_tiles(input_dir, output_path)
