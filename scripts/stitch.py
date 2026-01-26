import argparse
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

print("Starting stitch")
sys.stdout.flush()


def natural_sort_key(path: Path) -> list[int | str]:
    """Natural sorting key that handles numbers correctly."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.stem)]


def determine_grid_dimensions(tile_count: int, rows: int | None, cols: int | None) -> tuple[int, int]:
    """Determine grid dimensions for tiles."""
    if rows and cols:
        if rows * cols != tile_count:
            raise ValueError(f"Grid {rows}x{cols} doesn't match tile count {tile_count}")
        return rows, cols
    
    if rows:
        cols = tile_count // rows
        if rows * cols != tile_count:
            raise ValueError(f"Cannot arrange {tile_count} tiles in {rows} rows")
        return rows, cols
    
    if cols:
        rows = tile_count // cols
        if rows * cols != tile_count:
            raise ValueError(f"Cannot arrange {tile_count} tiles in {cols} columns")
        return rows, cols
    
    # Auto-detect square grid
    grid_size = int(np.sqrt(tile_count))
    if grid_size * grid_size != tile_count:
        raise ValueError(f"Tile count {tile_count} is not a perfect square. Specify --rows or --cols")
    return grid_size, grid_size


def stitch_tiles(
    input_dir: str,
    output_path: str,
    rows: int | None = None,
    cols: int | None = None,
    pattern: str = "tile_*.jpg",
) -> None:
    tiles: list[np.ndarray[Any, np.dtype[Any]]] = []

    # Find tiles matching pattern with natural sorting
    tile_files = sorted(
        [f for f in Path(input_dir).iterdir() if f.match(pattern)],
        key=natural_sort_key,
    )

    if not tile_files:
        raise ValueError(f"No tile files found matching pattern '{pattern}' in {input_dir}")

    for tile_file in tile_files:
        img = cv2.imread(str(tile_file))
        if img is None:
            raise ValueError(f"Could not load {tile_file}")
        tiles.append(img)

    tile_count = len(tiles)
    rows, cols = determine_grid_dimensions(tile_count, rows, cols)

    tiles_array = np.array(tiles)
    stitched = cv2.vconcat(
        [cv2.hconcat(list(tiles_row)) for tiles_row in np.array_split(tiles_array, rows)]
    )

    print(f"Stitched {tile_count} tiles in {rows}x{cols} grid, shape:", stitched.shape)
    print("About to write to", output_path)
    sys.stdout.flush()

    success = cv2.imwrite(output_path, stitched)
    if success:
        print("Stitch completed successfully")
    else:
        print("Failed to write output image")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch image tiles into a single image")
    parser.add_argument("input_dir", help="Directory containing tile images")
    parser.add_argument("output_path", help="Output image path")
    parser.add_argument("--rows", type=int, help="Number of rows in grid")
    parser.add_argument("--cols", type=int, help="Number of columns in grid")
    parser.add_argument(
        "--pattern", default="tile_*.jpg", help="File pattern for tiles (default: tile_*.jpg)"
    )

    args = parser.parse_args()
    stitch_tiles(args.input_dir, args.output_path, args.rows, args.cols, args.pattern)
