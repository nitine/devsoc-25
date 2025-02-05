import random
import tempfile
from pydantic import BaseModel, Field, validator
from sanic import HTTPResponse, Sanic, json
import httpx
import os
from dotenv import load_dotenv
import math
import structlog
import asyncio
import json as json_lib
from datetime import datetime
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sanic.exceptions import PayloadTooLarge

logger = structlog.get_logger()
app = Sanic("ElevationAPI")
app.config.REQUEST_TIMEOUT = 3000
app.config.RESPONSE_TIMEOUT = 3000
app.config.KEEP_ALIVE_TIMEOUT = 3000
app.config.REQUEST_MAX_SIZE = 100 * 1024 * 1024 * 20

# Increase file upload size limit (100MB)
app.config.REQUEST_BUFFER_QUEUE_SIZE = 100 * 1024 * 1024 * 100

# Load environment variables
load_dotenv()


class ElevationRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude between -90 and 90")
    lng: float = Field(
        ..., ge=-180, le=180, description="Longitude between -180 and 180"
    )
    area: float = Field(..., gt=0, description="Area in square kilometers")


class APIKeyManager:
    def __init__(self):
        self.api_keys = [
            os.getenv("GOOGLE_MAPS_API_KEY_1"),
            os.getenv("GOOGLE_MAPS_API_KEY_2"),
        ]
        self.current_index = 0
        self.lock = asyncio.Lock()
        self.request_counts = {
            key: 0 for key in self.api_keys if key
        }  # Only count valid keys
        self.last_reset = time.time()
        self.requests_per_minute = 6000  # Google Maps API limit per key

        # Log initial setup
        logger.info(f"Initialized APIKeyManager with {len(self.api_keys)} keys")

    async def get_next_key(self):
        async with self.lock:
            current_time = time.time()

            # Reset counters if a minute has passed
            if current_time - self.last_reset >= 60:
                logger.info("Resetting request counts for all keys")
                self.request_counts = {key: 0 for key in self.api_keys if key}
                self.last_reset = current_time

            # Try each key in rotation until finding one under the limit
            for _ in range(len(self.api_keys)):
                key = self.api_keys[self.current_index]
                if key and self.request_counts[key] < self.requests_per_minute:
                    self.request_counts[key] += 1

                    # Rotate to next key for next request
                    self.current_index = (self.current_index + 1) % len(self.api_keys)
                    return key

                # Move to next key if current one is at limit
                self.current_index = (self.current_index + 1) % len(self.api_keys)

            # If all keys are at limit, calculate wait time
            time_since_reset = current_time - self.last_reset
            time_to_wait = max(0, 60 - time_since_reset)

            if time_to_wait > 0:
                logger.warning(
                    f"All keys at limit. Waiting {time_to_wait:.2f} seconds for reset"
                )
                await asyncio.sleep(time_to_wait)
                self.last_reset = time.time()
                self.request_counts = {key: 0 for key in self.api_keys if key}
                return await self.get_next_key()

            # If we shouldn't wait, just reset and try again
            self.last_reset = current_time
            self.request_counts = {key: 0 for key in self.api_keys if key}
            return await self.get_next_key()


def save_to_json(data, lat, lng, area):
    """Save elevation data to a JSON file"""
    output_dir = "elevation_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"elevation.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json_lib.dump(data, f, indent=2)
    logger.info(f"Data saved to {filepath}")
    return filepath


def generate_denser_grid(center_lat, center_lng, area_km2, step_m=5):
    side_length_m = math.sqrt(area_km2) * 1000
    points_per_side = int(side_length_m / step_m) + 1
    start_lat = center_lat + (side_length_m / 2 / 111111)
    start_lng = center_lng - (
        side_length_m / 2 / (111111 * math.cos(math.radians(center_lat)))
    )
    points = []
    for i in range(points_per_side):
        for j in range(points_per_side):
            lat = start_lat - (i * step_m / 111111)
            lng = start_lng + (
                j * step_m / (111111 * math.cos(math.radians(center_lat)))
            )
            points.append((lat, lng))
    return points


async def process_batch(
    client, batch, batch_num, total_batches, key_manager, max_retries=5
):
    for attempt in range(max_retries):
        try:
            api_key = await key_manager.get_next_key()
            locations_str = "|".join(f"{lat:.7f},{lng:.7f}" for lat, lng in batch)
            url = "https://maps.googleapis.com/maps/api/elevation/json"
            params = {"locations": locations_str, "key": api_key}
            headers = {
                "Accept": "application/json",
                "User-Agent": "elevation-api-client",
            }
            response = await client.get(
                url, params=params, headers=headers, timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "OK" and data.get("results"):
                    results = []
                    for idx, result in enumerate(data["results"]):
                        point_lat, point_lng = batch[idx]
                        results.append(
                            {
                                "location": {
                                    "lat": point_lat,
                                    "lng": point_lng,
                                },
                                "elevation": result["elevation"],
                                "resolution": result.get("resolution", 0),
                            }
                        )
                    logger.info(f"Processed batch: {batch_num}/{total_batches}")
                    return results
                else:
                    logger.error(
                        f"API Error: {data.get('status')} - {data.get('error_message', 'No error message')}"
                    )
            else:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")

            if attempt < max_retries - 1:
                # Exponential backoff
                delay = (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Attempt {attempt + 1} failed. Retrying in {delay:.2f} seconds..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Max retries ({max_retries}) reached after {attempt + 1} attempts."
                )
                raise Exception("Max retries exceeded")
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {str(e)}")
            if attempt < max_retries - 1:
                # Exponential backoff
                delay = (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Attempt {attempt + 1} failed. Retrying in {delay:.2f} seconds..."
                )
                await asyncio.sleep(delay)
            else:
                raise


@app.middleware("response")
async def add_cors_headers(request, response):
    # Check if the response is an HTTPResponse (including error responses)
    if isinstance(response, HTTPResponse):
        # Add CORS headers to every response
        response.headers.update(
            {
                "Access-Control-Allow-Origin": "*",  # Configure this based on your needs
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": (
                    "origin, content-type, accept, "
                    "authorization, x-xsrf-token, x-request-id"
                ),
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Max-Age": "86400",
            }
        )
    return response


@app.get("/elevation-grid/<lat:float>/<lng:float>/<area:float>")
async def get_elevation_grid(request, lat: float, lng: float, area: float):
    try:
        key_manager = APIKeyManager()
        if not any(key_manager.api_keys):
            return json({"error": "No API keys configured"}, status=500)

        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return json({"error": "Invalid coordinates"}, status=400)

        points = generate_denser_grid(lat, lng, area, step_m=5)
        total_points = len(points)
        logger.info(f"Generated {total_points} points for {area} km2 area")
        # Larger batch size for speed
        batch_size = 400  # Close to Google's limit of 512
        batches = [
            points[i : i + batch_size] for i in range(0, len(points), batch_size)
        ]
        start_time = time.time()
        async with httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=100),
            http2=True,  # Enable HTTP/2 for better performance
        ) as client:
            # Process all batches concurrently
            tasks = [
                process_batch(client, batch, i + 1, len(batches), key_manager)
                for i, batch in enumerate(batches)
            ]
            all_results = await asyncio.gather(*tasks)

        elevations = []
        for batch_result in all_results:
            if batch_result:
                elevations.extend(batch_result)

        if not elevations:
            return json({"error": "Failed to get elevation data"}, status=500)

        end_time = time.time()
        processing_time = end_time - start_time
        response_data = {
            "processing_time": processing_time,
            "center": {"lat": lat, "lng": lng},
            "area_km2": area,
            "points_count": len(elevations),
            "points": elevations,
            "timestamp": datetime.now().isoformat(),
            "successful_points": len(elevations),
            "failed_points": total_points - len(elevations),
        }

        filepath = save_to_json(response_data, lat, lng, area)
        return json(response_data)

    except Exception as e:
        logger.exception("server_error", error=str(e))
        return json({"error": "Server error", "details": str(e)}, status=500)


# import json
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def count_json_objects(file_path: str) -> int:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json_lib.load(f)
            if isinstance(data, dict) and "points" in data:
                points = data["points"]
                if isinstance(points, list):
                    return len(points)
            return 0
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return 0


def build_up_b(rho, dt, dx, dy, u, v):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = rho * (
        1
        / dt
        * (
            (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
            + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
        )
        - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2
        - 2
        * (
            (u[2:, 1:-1] - u[0:-2, 1:-1])
            / (2 * dy)
            * (v[1:-1, 2:] - v[1:-1, 0:-2])
            / (2 * dx)
        )
        - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2
    )
    return b


def pressure_poisson(p, dx, dy, b, nit=50):
    pn = np.empty_like(p)
    for _ in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2
            + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2
        ) / (2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (
            2 * (dx**2 + dy**2)
        ) * b[
            1:-1, 1:-1
        ]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:, 0] = p[:, 1]
        p[-1, :] = 0
    return p


def cavity_flow(
    terrain,
    rho=1.225,
    nu=0.1,
    dt=0.001,
    nit=50,
    wind_speed=9.722,
    wind_direction=225.0,
    grid_size=100,
):
    dx = 2.0 / (grid_size - 1)
    dy = 2.0 / (grid_size - 1)

    direction_rad = math.radians(wind_direction)
    u_initial = -wind_speed * math.sin(direction_rad)
    v_initial = -wind_speed * math.cos(direction_rad)

    u = np.zeros((grid_size, grid_size))
    v = np.zeros((grid_size, grid_size))
    p = np.zeros((grid_size, grid_size))
    b = np.zeros((grid_size, grid_size))

    u[:, :] = u_initial
    v[:, :] = v_initial

    obstacle = np.where(terrain > np.percentile(terrain, 75), 1, 0)

    for _ in range(1000):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(rho, dt, dx, dy, u, v)
        p = pressure_poisson(p, dx, dy, b, nit)

        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
            - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
            - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])
            + nu
            * (
                dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                + dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
            )
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])
            - vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])
            - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])
            + nu
            * (
                dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])
                + dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])
            )
        )

        u[obstacle == 1] = 0
        v[obstacle == 1] = 0
        u[0, :] = u_initial
        v[:, 0] = v_initial

    return u, v, p


def analyze_terrain(file_path, wind_config):
    try:
        with open(file_path, "r") as f:
            data = json_lib.load(f)
            points = data["points"]
            lats = np.array([p["location"]["lat"] for p in points])
            lngs = np.array([p["location"]["lng"] for p in points])
            elevations = np.array([p["elevation"] for p in points])

        num_points = count_json_objects(file_path)
        grid_size = int(round(np.sqrt(num_points))) if num_points > 0 else 100
        grid_size = max(grid_size, 10)  # Ensure minimum grid size

        xi = np.linspace(lngs.min(), lngs.max(), grid_size)
        yi = np.linspace(lats.min(), lats.max(), grid_size)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((lngs, lats), elevations, (X, Y), method="cubic")
        terrain = (Z - Z.min()) / (Z.max() - Z.min())

        wind_speed_kmph = float(wind_config["average_wind_speed"])
        wind_speed_mps = wind_speed_kmph / 3.6
        wind_direction = float(wind_config["average_wind_direction"])

        u, v, p = cavity_flow(
            terrain,
            wind_speed=wind_speed_mps,
            wind_direction=wind_direction,
            grid_size=grid_size,
        )

        speed = np.sqrt(u**2 + v**2)
        power_density = 0.5 * 1.225 * speed**3
        turbulence = np.sqrt(np.gradient(u, axis=0) ** 2 + np.gradient(v, axis=1) ** 2)
        suitability = power_density / (1 + 5 * turbulence)

        optimal = np.argpartition(suitability.flatten(), -4)[-4:]
        i, j = np.unravel_index(optimal, suitability.shape)

        print("\nOptimal Wind Turbine Locations (Longitude, Latitude):")
        results = []
        for idx in range(4):
            lat = Y[i[idx], j[idx]]
            lng = X[i[idx], j[idx]]
            elev = Z[i[idx], j[idx]]
            wind_speed = speed[i[idx], j[idx]]
            results.append(
                {
                    "longitude": round(lng, 6),
                    "latitude": round(lat, 6),
                    "elevation": round(elev, 1),
                    "predicted_wind_speed": round(wind_speed, 2),
                }
            )
            print(
                f"Location {idx+1}: ({lng:.6f}, {lat:.6f}), Elevation: {elev:.1f}m, Wind Speed: {wind_speed:.2f} m/s"
            )

        return results

    except Exception as e:
        print(f"Error: {str(e)}")
        return []


# @app.route("/analyze-terrain", methods=["POST", "OPTIONS"])
# async def analyze_wind(request):
#     # Handle OPTIONS request for CORS preflight
#     if request.method == "OPTIONS":
#         return json(
#             {"status": "ok"},
#             headers={
#                 "Access-Control-Allow-Origin": "*",
#                 "Access-Control-Allow-Methods": "POST, OPTIONS",
#                 "Access-Control-Allow-Headers": "Content-Type, Authorization",
#                 "Access-Control-Max-Age": "86400",
#             },
#         )
#     try:
#         if request.content_type != "application/json":
#             return json(
#                 {
#                     "error": "Invalid content type",
#                     "details": "Content-Type must be application/json",
#                 },
#                 status=400,
#             )
#         try:
#             raw_data = request.body.decode("utf-8").strip()
#             data = json_lib.loads(raw_data)
#         except Exception as e:
#             logger.error("Failed parsing request body", error=str(e))
#             return json(
#                 {
#                     "error": "Invalid JSON format",
#                     "details": "Ensure the request body is valid JSON without trailing data",
#                 },
#                 status=400,
#             )
#         # Process the data...
#         # (validate fields, perform analysis, etc.)
#         results = analyze_terrain(data.get("file_path"), data.get("wind_config"))
#         if not results:
#             return json(
#                 {
#                     "error": "Analysis failed",
#                     "details": "No results were generated from the analysis",
#                 },
#                 status=500,
#             )
#         return json(
#             {
#                 "status": "success",
#                 "results": results,
#                 "metadata": {
#                     "file_path": data.get("file_path"),
#                     "wind_config": data.get("wind_config"),
#                     "timestamp": datetime.now().isoformat(),
#                 },
#             }
#         )
#     except Exception as e:
#         logger.exception("server_error", error=str(e))
#         return json({"error": "Server error", "details": str(e)}, status=500)


@app.route("/analyze-terrain", methods=["GET", "POST", "OPTIONS"])
async def analyze_wind(request):
    # Common CORS headers
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS, POST",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age": "86400",
    }

    # Handle CORS preflight
    if request.method == "OPTIONS":
        return json({"status": "ok"}, headers=headers)

    # Your existing logic
    test_payload = {
        "file_path": "elevation_data/elevation.json",
        "wind_config": {
            "average_wind_speed": 15,
            "average_wind_direction": 0,
        },
    }

    print("Using hard-coded payload:", test_payload)
    results = analyze_terrain(test_payload["file_path"], test_payload["wind_config"])

    if not results:
        return json(
            {
                "error": "Analysis failed",
                "details": "No results were generated from the analysis",
            },
            status=500,
            headers=headers,  # Include CORS headers in error response
        )

    response_data = {
        "results": results,
    }

    # Include CORS headers in successful response
    return json(response_data, headers=headers)


@app.route("/elevation", methods=["POST", "OPTIONS"])
async def post_elevation_grid(request):
    # Handle OPTIONS request for CORS preflight
    if request.method == "OPTIONS":
        return json(
            {"status": "ok"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400",  # 24 hours
            },
        )
    try:
        try:
            data = request.json
            elevation_request = ElevationRequest(**data)
        except ValueError as e:
            return {
                "status": "error",
                "code": 400,
                "message": "Invalid request data",
                "details": str(e),
            }

        lat = elevation_request.lat
        lng = elevation_request.lng
        area = elevation_request.area
        key_manager = APIKeyManager()
        if not any(key_manager.api_keys):
            return json({"error": "No API keys configured"}, status=500)

        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return json({"error": "Invalid coordinates"}, status=400)

        points = generate_denser_grid(lat, lng, area, step_m=25)
        total_points = len(points)
        logger.info(f"Generated {total_points} points for {area} km2 area")
        # Larger batch size for speed
        batch_size = 400  # Close to Google's limit of 512
        batches = [
            points[i : i + batch_size] for i in range(0, len(points), batch_size)
        ]
        start_time = time.time()
        async with httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=100),
            http2=True,  # Enable HTTP/2 for better performance
        ) as client:
            # Process all batches concurrently
            tasks = [
                process_batch(client, batch, i + 1, len(batches), key_manager)
                for i, batch in enumerate(batches)
            ]
            all_results = await asyncio.gather(*tasks)

        elevations = []
        for batch_result in all_results:
            if batch_result:
                elevations.extend(batch_result)

        if not elevations:
            return json({"error": "Failed to get elevation data"}, status=500)

        end_time = time.time()
        processing_time = end_time - start_time
        response_data = {
            "processing_time": processing_time,
            "center": {"lat": lat, "lng": lng},
            "area_km2": area,
            "points_count": len(elevations),
            "points": elevations,
            "timestamp": datetime.now().isoformat(),
            "successful_points": len(elevations),
            "failed_points": total_points - len(elevations),
            "step_m": 5,
        }

        filepath = save_to_json(response_data, lat, lng, area)
        response_data["file_saved"] = filepath
        return json(response_data)

    except Exception as e:
        logger.exception("server_error", error=str(e))
        return json({"error": "Server error", "details": str(e)}, status=500)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True,
        auto_reload=True,
    )
