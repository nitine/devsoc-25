from sanic import Sanic, json
from sanic.response import JSONResponse
import httpx
import os
from dotenv import load_dotenv
import math
import structlog
import asyncio
import json as json_lib
from datetime import datetime
import time

# Configure structlog
logger = structlog.get_logger()
app = Sanic("ElevationAPI")
app.config.REQUEST_TIMEOUT = 3000
app.config.RESPONSE_TIMEOUT = 3000
app.config.KEEP_ALIVE_TIMEOUT = 3000

# Load environment variables
load_dotenv()


class APIKeyManager:
    def __init__(self):
        self.api_keys = [
            os.getenv("GOOGLE_MAPS_API_KEY_1"),
            os.getenv("GOOGLE_MAPS_API_KEY_2"),
        ]
        self.current_index = 0
        self.lock = asyncio.Lock()
        self.request_counts = {key: 0 for key in self.api_keys}
        self.last_reset = time.time()

    async def get_next_key(self):
        async with self.lock:
            # Reset counters if a minute has passed
            current_time = time.time()
            if current_time - self.last_reset >= 60:
                self.request_counts = {key: 0 for key in self.api_keys}
                self.last_reset = current_time

            # Get next available key
            for _ in range(len(self.api_keys)):
                key = self.api_keys[self.current_index]
                if self.request_counts[key] < 6000:  # 6000 requests per minute limit
                    self.request_counts[key] += 1
                    self.current_index = (self.current_index + 1) % len(self.api_keys)
                    return key
                self.current_index = (self.current_index + 1) % len(self.api_keys)

            # If all keys are at limit, wait and try again
            await asyncio.sleep(1)
            return await self.get_next_key()


def save_to_json(data, lat, lng, area):
    """Save elevation data to a JSON file"""
    output_dir = "elevation_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"elevation_{lat}_{lng}_{area}km2_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json_lib.dump(data, f, indent=2)
    logger.info(f"Data saved to {filepath}")
    return filepath


def generate_grid_points(center_lat, center_lng, area_km2, step_m=15):
    """Generate points in a square grid"""
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
    client, batch, batch_num, total_batches, key_manager, max_retries=3
):
    """Process a single batch with enhanced error handling"""
    for retry in range(max_retries):
        try:
            api_key = await key_manager.get_next_key()

            # Format locations properly
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

            if response.status_code != 200:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
                await asyncio.sleep(1)
                continue

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
                if data.get("status") == "OVER_QUERY_LIMIT":
                    await asyncio.sleep(2)
                continue

        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {str(e)}")
            await asyncio.sleep(1)
            continue
    return []


async def process_batches_with_rate_limit(client, batches, key_manager):
    """Process batches with optimized rate limiting"""
    concurrent_requests = len(key_manager.api_keys) * 50
    results = []

    for i in range(0, len(batches), concurrent_requests):
        current_batches = batches[i : i + concurrent_requests]
        tasks = [
            process_batch(client, batch, i + j + 1, len(batches), key_manager)
            for j, batch in enumerate(current_batches)
        ]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        if i + concurrent_requests < len(batches):
            await asyncio.sleep(1)

    return results


@app.get("/elevation-grid/<lat:float>/<lng:float>/<area:float>")
async def get_elevation_grid(request, lat: float, lng: float, area: float):
    try:
        key_manager = APIKeyManager()
        if not any(key_manager.api_keys):
            return json({"error": "No API keys configured"}, status=500)

        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return json({"error": "Invalid coordinates"}, status=400)

        points = generate_grid_points(lat, lng, area, step_m=15)
        total_points = len(points)

        batch_size = 250
        batches = [
            points[i : i + batch_size] for i in range(0, len(points), batch_size)
        ]

        start_time = time.time()

        async with httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=40, max_connections=40),
        ) as client:
            all_results = await process_batches_with_rate_limit(
                client, batches, key_manager
            )

        end_time = time.time()
        processing_time = end_time - start_time

        elevations = []
        for batch_result in all_results:
            if batch_result:
                elevations.extend(batch_result)

        if not elevations:
            return json({"error": "Failed to get elevation data"}, status=500)

        response_data = {
            "processing_time_seconds": processing_time,
            "center": {"lat": lat, "lng": lng},
            "area_km2": area,
            "step_m": 15,
            "points_count": len(elevations),
            "points": elevations,
            "timestamp": datetime.now().isoformat(),
            "successful_points": len(elevations),
            "failed_points": total_points - len(elevations),
            "performance_metrics": {
                "points_per_second": (
                    len(elevations) / processing_time if processing_time > 0 else 0
                ),
                "total_batches": len(batches),
                "concurrent_requests": len(key_manager.api_keys) * 25,
            },
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
