

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
    sample_step: int = 20,
) -> list[dict]:
    """Estimate forest presence in H3 cells from a GeoTIFF."""

    import rasterio
    from rasterio.windows import from_bounds

    if not (-180 <= west < east <= 180):
        raise ValueError("Invalid longitude bounds.")

    if not (-90 <= south < north <= 90):
        raise ValueError("Invalid latitude bounds.")

    if sample_step < 1:
        raise ValueError("sample_step must be at least 1.")

    with rasterio.open(raster_path) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError("Raster must use EPSG:4326.")

        window = from_bounds(
            west, south, east, north, src.transform
        )
        window = window.round_offsets().round_lengths()

        full_window = rasterio.windows.Window(
            0, 0, src.width, src.height
        )
        window = window.intersection(full_window)

        if window.width <= 0 or window.height <= 0:
            return []

        height = int(window.height)
        width = int(window.width)

        rows = list(range(0, height, sample_step))
        cols = list(range(0, width, sample_step))

        cells = {}

        for row in rows:
            for col in cols:
                pixel = src.read(
                    1,
                    window=rasterio.windows.Window(
                        int(window.col_off) + col,
                        int(window.row_off) + row,
                        1,
                        1,
                    ),
                )[0, 0]

                if pixel != 1:
                    continue

                lon, lat = rasterio.transform.xy(
                    src.window_transform(window),
                    row,
                    col,
                )

                cell_id = h3.latlng_to_cell(
                    lat, lon, resolution
                )

                if cell_id not in cells:
                    center_lat, center_lon = (
                        h3.cell_to_latlng(cell_id)
                    )

                    cells[cell_id] = {
                        "cell_id": cell_id,
                        "center_latitude": center_lat,
                        "center_longitude": center_lon,
                        "resolution": resolution,
                        "sampled_forest_blocks": 0,
                    }

                cells[cell_id]["sampled_forest_blocks"] += 1

    return list(cells.values())
