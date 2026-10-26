

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
