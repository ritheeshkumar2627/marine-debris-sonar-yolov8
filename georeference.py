"""
georeference.py
================
Geospatial coordinate converter for Side-Scan Sonar (SSS) and Marine Debris surveys.

Converts sonar waterfall detections (X_pixel, Y_pixel) into geographic WGS-84
Latitude and Longitude coordinates using:
1. Raw .XTF acoustic ping navigation headers (pyxtf)
2. GeoTIFF geospatial transforms (rasterio)
3. GPS track log interpolation (timestamp/ping CSV)
4. Spherical forward geodesy based on vessel position, heading, and across-track range.
"""

import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

EARTH_RADIUS_M = 6378137.0  # WGS-84 equatorial radius in meters


def calculate_target_gps(
    vessel_lat: float,
    vessel_lon: float,
    vessel_heading_deg: float,
    across_track_m: float
) -> Tuple[float, float]:
    """
    Computes real-world target GPS coordinates from vessel position, heading, and across-track offset.

    Parameters:
    -----------
    vessel_lat : float
        Vessel latitude in decimal degrees.
    vessel_lon : float
        Vessel longitude in decimal degrees.
    vessel_heading_deg : float
        Vessel heading (course over ground) in degrees (0 = North, 90 = East, etc.).
    across_track_m : float
        Distance in meters perpendicular to vessel track:
        Positive (+) = Starboard (Right side of vessel)
        Negative (-) = Port (Left side of vessel)

    Returns:
    --------
    (target_lat, target_lon) : Tuple[float, float]
        Target latitude and longitude in decimal degrees (WGS-84).
    """
    if abs(across_track_m) < 1e-6:
        return vessel_lat, vessel_lon

    # Target bearing relative to true north
    if across_track_m >= 0:
        bearing_deg = (vessel_heading_deg + 90.0) % 360.0
    else:
        bearing_deg = (vessel_heading_deg - 90.0) % 360.0

    distance_m = abs(across_track_m)
    bearing_rad = math.radians(bearing_deg)
    lat1_rad = math.radians(vessel_lat)
    lon1_rad = math.radians(vessel_lon)
    angular_dist = distance_m / EARTH_RADIUS_M

    lat2_rad = math.asin(
        math.sin(lat1_rad) * math.cos(angular_dist) +
        math.cos(lat1_rad) * math.sin(angular_dist) * math.cos(bearing_rad)
    )

    lon2_rad = lon1_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_dist) * math.cos(lat1_rad),
        math.cos(angular_dist) - math.sin(lat1_rad) * math.sin(lat2_rad)
    )

    target_lat = math.degrees(lat2_rad)
    target_lon = math.degrees(lon2_rad)

    # Normalize longitude to [-180, 180]
    target_lon = (target_lon + 180.0) % 360.0 - 180.0

    return round(target_lat, 7), round(target_lon, 7)


def extract_xtf_ping_info(xtf_path: Path) -> List[Dict[str, float]]:
    """
    Extracts navigation metadata per ping packet from an .xtf sonar file.
    Returns list of dicts with: lat, lon, heading, slant_range_m.
    """
    try:
        import pyxtf
        header, packets = pyxtf.xtf_read(str(xtf_path))
        pings = []
        for pkt in packets:
            if hasattr(pkt, "ShipYpos") and hasattr(pkt, "ShipXpos"):
                lat = float(pkt.ShipYpos)
                lon = float(pkt.ShipXpos)
                heading = float(getattr(pkt, "SensorHeading", 0.0))
                slant_range = float(getattr(pkt, "SlantRange", 50.0))
                pings.append({
                    "lat": lat,
                    "lon": lon,
                    "heading": heading,
                    "slant_range": slant_range
                })
        return pings
    except Exception as e:
        print(f"[!] Warning: Could not read .xtf navigation packets: {e}")
        return []


def read_geotiff_coords(tif_path: Path, x_px: float, y_px: float) -> Optional[Tuple[float, float]]:
    """
    Extracts geographic coordinates (lat, lon) from a GeoTIFF using rasterio if available.
    """
    try:
        import rasterio
        from rasterio.warp import transform
        with rasterio.open(str(tif_path)) as src:
            x_geo, y_geo = src.xy(y_px, x_px)
            if src.crs and not src.crs.is_epsg_code(4326):
                xs, ys = transform(src.crs, "EPSG:4326", [x_geo], [y_geo])
                return round(ys[0], 7), round(xs[0], 7)
            return round(y_geo, 7), round(x_geo, 7)
    except Exception:
        return None


def pixel_to_gps(
    x_px: float,
    y_px: float,
    image_width: int,
    image_height: int,
    vessel_lat: Optional[float] = None,
    vessel_lon: Optional[float] = None,
    vessel_heading: float = 0.0,
    max_range_m: float = 50.0,
    nav_pings: Optional[List[Dict[str, float]]] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Converts sonar pixel (x_px, y_px) to GPS coordinates (lat, lon).

    Across-track geometry:
    - Sonar Nadir (directly beneath boat) is at horizontal center (image_width / 2.0).
    - x_px < center: Port side (across_track < 0)
    - x_px > center: Starboard side (across_track > 0)
    """
    # If ping packets are available from .xtf
    if nav_pings and len(nav_pings) > 0:
        ping_idx = int(min(max(0, y_px), len(nav_pings) - 1))
        ping = nav_pings[ping_idx]
        vessel_lat = ping.get("lat")
        vessel_lon = ping.get("lon")
        vessel_heading = ping.get("heading", vessel_heading)
        max_range_m = ping.get("slant_range", max_range_m)

    if vessel_lat is None or vessel_lon is None:
        return None, None

    half_w = image_width / 2.0
    if half_w <= 0:
        half_w = 1.0
    norm_offset = (x_px - half_w) / half_w
    across_track_m = norm_offset * max_range_m

    return calculate_target_gps(vessel_lat, vessel_lon, vessel_heading, across_track_m)


def export_geojson(detections: List[Dict[str, Any]], output_path: Path):
    """
    Exports a list of detections with geographic coordinates to standard GeoJSON (RFC 7946).
    Can be loaded directly into QGIS, ArcGIS, Google Earth, or geojson.io.
    """
    features = []
    for d in detections:
        lat = d.get("latitude")
        lon = d.get("longitude")
        if lat is None or lon is None:
            continue

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "target_id": d.get("target_id"),
                "class_name": d.get("class_name"),
                "confidence": d.get("confidence"),
                "x_pixel": d.get("x_pixel"),
                "y_pixel": d.get("y_pixel"),
                "width_px": d.get("width_px"),
                "height_px": d.get("height_px")
            }
        }
        features.append(feature)

    geojson_obj = {
        "type": "FeatureCollection",
        "name": "Marine_Debris_Sonar_Detections",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f, indent=2)

    print(f"[+] GeoJSON map saved to: {output_path} ({len(features)} georeferenced targets)")
