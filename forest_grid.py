

from typing import Any

import h3


def generate_grid(
    latitude: float,
    longitude: float,
    resolution: int = 5,
) -> dict[str, Any]:
    """Generate the H3 cell containing a geographic point."""

    cell_id = h3.latlng_to_cell(
        latitude,
        longitude,
        resolution,
    )

    boundary = h3.cell_to_boundary(cell_id)

    return {
        "cell_id": cell_id,
        "center_latitude": latitude,
        "center_longitude": longitude,
        "resolution": resolution,
        "boundary": [
            {"latitude": lat, "longitude": lon}
            for lat, lon in boundary
        ],
    }


def generate_nearby_grid(
    latitude: float,
    longitude: float,
    resolution: int = 5,
    ring_size: int = 2,
) -> list[dict[str, Any]]:
    """Generate a group of neighboring cells around a point."""

    center_cell = h3.latlng_to_cell(
        latitude,
        longitude,
        resolution,
    )

    cells = h3.grid_disk(center_cell, ring_size)

    results = []

    for cell_id in cells:
        center_lat, center_lon = h3.cell_to_latlng(cell_id)
        boundary = h3.cell_to_boundary(cell_id)

        results.append({
            "cell_id": cell_id,
            "center_latitude": center_lat,
            "center_longitude": center_lon,
            "resolution": resolution,
            "boundary": [
                {"latitude": lat, "longitude": lon}
                for lat, lon in boundary
            ],
        })

    return results
    
from pathlib import Path

import rasterio
from shapely.geometry import Polygon, box


def identify_forest_cells(
    raster_path: str,
    west: float,
    south: float,
    east: float,
    north: float,
    resolution: int = 5,
) -> list[dict]:
    """
    Identify H3 cells containing forest pixels in a local
    GeoTIFF forest-cover tile.

    The raster must use WGS84 coordinates (EPSG:4326).
    The function expects forest pixels to have value 1.
    """

    if not Path(raster_path).is_file():
        raise FileNotFoundError(
            f"Forest-cover raster not found: {raster_path}"
        )

    if not (-180 <= west < east <= 180):
        raise ValueError("Invalid longitude bounds.")

    if not (-90 <= south < north <= 90):
        raise ValueError("Invalid latitude bounds.")

    with rasterio.open(raster_path) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(
                "The raster must use WGS84 (EPSG:4326)."
            )

        window = rasterio.windows.from_bounds(
            west, south, east, north, src.transform
        )

        window = window.round_offsets().round_lengths()
        window = window.intersection(
            rasterio.windows.Window(
                0, 0, src.width, src.height
            )
        )

        if window.width <= 0 or window.height <= 0:
            return []

        data = src.read(1, window=window)
        transform = src.window_transform(window)

    forest_rows, forest_cols = (data == 1).nonzero()
    cells = {}

    for row, col in zip(forest_rows, forest_cols):
        lon, lat = rasterio.transform.xy(
            transform, int(row), int(col)
        )

        cell_id = h3.latlng_to_cell(lat, lon, resolution)

        if cell_id not in cells:
            cells[cell_id] = {
                "cell_id": cell_id,
                "center_latitude": h3.cell_to_latlng(
                    cell_id
                )[0],
                "center_longitude": h3.cell_to_latlng(
                    cell_id
                )[1],
                "resolution": resolution,
                "forest_pixels_sampled": 0,
            }

        cells[cell_id]["forest_pixels_sampled"] += 1

    return list(cells.values())

