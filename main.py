import random
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

logger = structlog.get_logger()
app = Sanic("ElevationAPI")
app.config.REQUEST_TIMEOUT = 3000
app.config.RESPONSE_TIMEOUT = 3000
app.config.KEEP_ALIVE_TIMEOUT = 3000

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
    filename = f"elevation_{lat}_{lng}_{area}km2_{timestamp}.json"
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
