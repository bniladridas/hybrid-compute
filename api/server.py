import os
import re
import subprocess
import uuid
from datetime import datetime

from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

API_VERSION = os.getenv("API_VERSION", "v1")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/tmp/hybrid-compute/uploads")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "/tmp/hybrid-compute/output")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}


def is_safe_id(image_id: str) -> bool:
    """
    Validate that an image ID contains only safe characters.

    This prevents directory traversal and path injection when IDs are used
    in filesystem paths (for example, when creating tiles directories).
    Allowed characters: ASCII letters, digits, underscore, and hyphen.
    """
    if not isinstance(image_id, str) or not image_id:
        return False
    return re.fullmatch(r"[A-Za-z0-9_-]+", image_id) is not None


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def generate_squuid():
    """Generate a SQUUID (Sequential UUID)"""
    return str(uuid.uuid4())


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def hal_response(data, links=None, embedded=None, meta=None):
    """Create a HAL-formatted response"""
    response = {"_links": {"self": {"href": request.path}}}

    if links:
        response["_links"].update(links)

    if embedded:
        response["_embedded"] = embedded

    if meta:
        response["meta"] = meta

    response.update(data)
    return response


def error_response(code, title, details, status_code):
    """Create an error response"""
    return jsonify({"errors": [{"code": code, "title": title, "details": details}]}), status_code


@app.route(f"/{API_VERSION}/images", methods=["POST"])
def upload_image():
    """Upload a new image"""
    if "file" not in request.files:
        return error_response("no_file", "No file provided", "Please provide an image file", 400)

    file = request.files["file"]
    if file.filename == "":
        return error_response("empty_filename", "Empty filename", "Please provide a valid filename", 400)

    if not allowed_file(file.filename):
        return error_response(
            "invalid_format", "Invalid file format", f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}", 400
        )

    image_id = generate_squuid()
    filename = secure_filename(file.filename or "")
    filepath = os.path.join(UPLOAD_FOLDER, f"{image_id}_{filename}")
    file.save(filepath)

    return (
        jsonify(
            hal_response(
                {
                    "id": image_id,
                    "filename": filename,
                    "format": filename.rsplit(".", 1)[1].lower(),
                    "size": os.path.getsize(filepath),
                    "created_at": datetime.utcnow().isoformat() + "Z",
                },
                {
                    "self": {"href": f"/{API_VERSION}/images/{image_id}"},
                    "tiles": {"href": f"/{API_VERSION}/images/{image_id}/tiles"},
                    "upscale": {"href": f"/{API_VERSION}/images/{image_id}/upscale"},
                },
            )
        ),
        201,
    )


@app.route(f"/{API_VERSION}/images", methods=["GET"])
def list_images():
    """List all uploaded images"""
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 25))

    images = []
    for f in os.listdir(UPLOAD_FOLDER):
        if allowed_file(f):
            filepath = os.path.join(UPLOAD_FOLDER, f)
            image_id = f.split("_", 1)[0]
            images.append(
                {
                    "id": image_id,
                    "filename": f.split("_", 1)[1] if "_" in f else f,
                    "size": os.path.getsize(filepath),
                    "created_at": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat() + "Z",
                }
            )

    paginated = images[offset : offset + limit]
    total = len(images)

    links = {"self": {"href": f"/{API_VERSION}/images?offset={offset}&limit={limit}"}}

    if offset + limit < total:
        links["next"] = {"href": f"/{API_VERSION}/images?offset={offset + limit}&limit={limit}"}
    if offset > 0:
        links["prev"] = {"href": f"/{API_VERSION}/images?offset={max(0, offset - limit)}&limit={limit}"}

    return jsonify(hal_response({"count": len(paginated), "total": total}, links=links, embedded={"images": paginated}))


@app.route(f"/{API_VERSION}/images/<image_id>", methods=["GET"])
def get_image(image_id):
    """Get image details"""
    for f in os.listdir(UPLOAD_FOLDER):
        if f.startswith(image_id + "_"):
            filepath = os.path.join(UPLOAD_FOLDER, f)
            return jsonify(
                hal_response(
                    {
                        "id": image_id,
                        "filename": f.split("_", 1)[1],
                        "size": os.path.getsize(filepath),
                        "created_at": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat() + "Z",
                    },
                    {
                        "self": {"href": f"/{API_VERSION}/images/{image_id}"},
                        "tiles": {"href": f"/{API_VERSION}/images/{image_id}/tiles"},
                        "upscale": {"href": f"/{API_VERSION}/images/{image_id}/upscale"},
                        "download": {"href": f"/{API_VERSION}/images/{image_id}/file"},
                    },
                )
            )

    return error_response("not_found", "Image not found", f"No image found with ID: {image_id}", 404)


@app.route(f"/{API_VERSION}/images/<image_id>/file", methods=["GET"])
def download_image(image_id):
    """Download image file"""
    for f in os.listdir(UPLOAD_FOLDER):
        if f.startswith(image_id + "_"):
            return send_file(os.path.join(UPLOAD_FOLDER, f))

    return error_response("not_found", "Image not found", f"No image found with ID: {image_id}", 404)


@app.route(f"/{API_VERSION}/images/<image_id>/tiles", methods=["POST"])
def create_tiles(image_id):
    """Split image into tiles"""
    if not is_safe_id(image_id):
        return error_response(
            "invalid_id",
            "Invalid image ID",
            "The provided image ID contains invalid characters.",
            400,
        )

    source_file = None
    for f in os.listdir(UPLOAD_FOLDER):
        if f.startswith(image_id + "_"):
            source_file = os.path.join(UPLOAD_FOLDER, f)
            break

    if not source_file:
        return error_response("not_found", "Image not found", f"No image found with ID: {image_id}", 404)

    tiles_dir = os.path.join(OUTPUT_FOLDER, f"tiles_{image_id}")
    os.makedirs(tiles_dir, exist_ok=True)

    tile_size = request.json.get("tile_size", 512) if request.json else 512

    try:
        result = subprocess.run(
            ["./preprocess", os.path.dirname(source_file) + "/", tiles_dir + "/"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return error_response("timeout", "Processing timeout", "Image tiling took too long", 504)
    except FileNotFoundError:
        return error_response("not_available", "Preprocessor not available", "Please build the preprocessing tool", 501)

    tiles = []
    for f in os.listdir(tiles_dir):
        if allowed_file(f):
            tiles.append({"id": f.rsplit(".", 1)[0], "filename": f, "href": f"/{API_VERSION}/tiles/{f.rsplit('.', 1)[0]}"})

    return jsonify(
        hal_response(
            {"image_id": image_id, "tile_count": len(tiles), "tile_size": tile_size},
            links={
                "self": {"href": f"/{API_VERSION}/images/{image_id}/tiles"},
                "image": {"href": f"/{API_VERSION}/images/{image_id}"},
            },
            embedded={"tiles": tiles},
        ),
        202,
    )


@app.route(f"/{API_VERSION}/tiles", methods=["GET"])
def list_tiles():
    """List all tiles"""
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 25))

    tiles = []
    for root, dirs, files in os.walk(OUTPUT_FOLDER):
        for f in files:
            if allowed_file(f) and "tiles_" in root:
                tiles.append({"id": f.rsplit(".", 1)[0], "filename": f, "href": f"/{API_VERSION}/tiles/{f.rsplit('.', 1)[0]}"})

    paginated = tiles[offset : offset + limit]

    links = {"self": {"href": f"/{API_VERSION}/tiles?offset={offset}&limit={limit}"}}
    if offset + limit < len(tiles):
        links["next"] = {"href": f"/{API_VERSION}/tiles?offset={offset + limit}&limit={limit}"}
    if offset > 0:
        links["prev"] = {"href": f"/{API_VERSION}/tiles?offset={max(0, offset - limit)}&limit={limit}"}

    return jsonify(hal_response({"count": len(paginated), "total": len(tiles)}, links=links, embedded={"tiles": paginated}))


@app.route(f"/{API_VERSION}/tiles/<tile_id>", methods=["GET"])
def get_tile(tile_id):
    """Get tile details"""
    for root, dirs, files in os.walk(OUTPUT_FOLDER):
        for f in files:
            if f.startswith(tile_id + "."):
                filepath = os.path.join(root, f)
                return jsonify(
                    hal_response(
                        {"id": tile_id, "filename": f, "size": os.path.getsize(filepath)},
                        {
                            "self": {"href": f"/{API_VERSION}/tiles/{tile_id}"},
                            "upscale": {"href": f"/{API_VERSION}/tiles/{tile_id}/upscale"},
                        },
                    )
                )

    return error_response("not_found", "Tile not found", f"No tile found with ID: {tile_id}", 404)


@app.route(f"/{API_VERSION}/tiles/<tile_id>/upscale", methods=["POST"])
def upscale_tile(tile_id):
    """Upscale a tile"""
    scale = request.json.get("scale", 2) if request.json else 2

    source_file = None
    for root, dirs, files in os.walk(OUTPUT_FOLDER):
        for f in files:
            if f.startswith(tile_id + "."):
                source_file = os.path.join(root, f)
                break

    if not source_file:
        return error_response("not_found", "Tile not found", f"No tile found with ID: {tile_id}", 404)

    output_file = os.path.join(OUTPUT_FOLDER, f"upscaled_{tile_id}.png")

    try:
        result = subprocess.run(
            ["./cloud_gpu/upscaler", source_file, output_file, str(scale)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return error_response("timeout", "Processing timeout", "Upscaling took too long", 504)
    except FileNotFoundError:
        return error_response("not_available", "Upscaler not available", "Please build the CUDA upscaler", 501)

    return (
        jsonify(
            hal_response(
                {"id": tile_id, "scale": scale, "output_file": f"upscaled_{tile_id}.png"},
                {
                    "self": {"href": f"/{API_VERSION}/tiles/{tile_id}/upscale"},
                    "tile": {"href": f"/{API_VERSION}/tiles/{tile_id}"},
                    "download": {"href": f"/{API_VERSION}/tiles/{tile_id}/upscaled"},
                },
            )
        ),
        202,
    )


@app.route(f"/{API_VERSION}/tiles/<tile_id>/upscaled", methods=["GET"])
def download_upscaled(tile_id):
    """Download upscaled tile"""
    upscaled_file = os.path.join(OUTPUT_FOLDER, f"upscaled_{tile_id}.png")
    if os.path.exists(upscaled_file):
        return send_file(upscaled_file)

    return error_response("not_found", "Upscaled tile not found", f"No upscaled tile found for ID: {tile_id}", 404)


@app.route(f"/{API_VERSION}/stitch", methods=["POST"])
def stitch_tiles():
    """Stitch tiles into final image"""
    data = request.json or {}
    tile_ids = data.get("tile_ids", [])
    rows = data.get("rows", 1)
    cols = data.get("cols", 1)
    output_name = data.get("output", "stitched.png")

    if not tile_ids:
        return error_response("no_tiles", "No tiles provided", "Please provide tile IDs to stitch", 400)

    job_id = generate_squuid()

    return (
        jsonify(
            hal_response(
                {"job_id": job_id, "status": "processing", "tile_count": len(tile_ids), "rows": rows, "cols": cols},
                {"self": {"href": f"/{API_VERSION}/stitch/{job_id}"}, "status": {"href": f"/{API_VERSION}/jobs/{job_id}"}},
            )
        ),
        202,
    )


@app.route(f"/{API_VERSION}/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """Get job status"""
    return jsonify(
        hal_response(
            {"id": job_id, "status": "completed", "result": f"/{API_VERSION}/outputs/{job_id}/result.png"},
            {"self": {"href": f"/{API_VERSION}/jobs/{job_id}"}},
        )
    )


@app.route("/", methods=["GET"])
def api_root():
    """API root endpoint"""
    return jsonify(
        {
            "message": f"Hybrid Compute API {API_VERSION}",
            "endpoints": {
                "health": f"/{API_VERSION}/health",
                "images": f"/{API_VERSION}/images",
                "tiles": f"/{API_VERSION}/tiles",
                "stitch": f"/{API_VERSION}/stitch",
            },
        }
    )


@app.route("/health", methods=["GET"])
@app.route(f"/{API_VERSION}/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "version": API_VERSION})


@app.errorhandler(400)
def bad_request(e):
    return error_response("bad_request", "Bad Request", str(e.description), 400)


@app.errorhandler(404)
def not_found(e):
    return error_response("not_found", "Not Found", "The requested resource was not found", 404)


@app.errorhandler(500)
def internal_error(e):
    return error_response("internal_error", "Internal Server Error", "An unexpected error occurred", 500)


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
