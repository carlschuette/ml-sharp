
import logging
import shutil
import uuid
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn.functional as F
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Import sharp modules
from sharp.models import (
    PredictorParams,
    create_predictor,
)
from sharp.utils import io
from sharp.utils.gaussians import (
    save_ply,
    unproject_gaussians,
)
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "sharp_model.pt"

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
CONVERT_SCRIPT = FRONTEND_DIR / "convert_to_ksplat.js"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Global variables
device = "cpu"
gaussian_predictor = None
status_queues: Dict[str, asyncio.Queue] = {}
inference_lock = asyncio.Lock()

def get_device():
    global device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    LOGGER.info(f"Using device: {device}")
    return device

def load_model():
    global gaussian_predictor
    if gaussian_predictor is not None:
        return

    device_name = get_device()
    
    if not CHECKPOINT_PATH.exists():
        LOGGER.info(f"Downloading default model from {DEFAULT_MODEL_URL}")
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True, map_location=device_name)
    else:
        LOGGER.info(f"Loading checkpoint from {CHECKPOINT_PATH}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device_name, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device_name)
    LOGGER.info("Model loaded successfully")

@app.on_event("startup")
async def startup_event():
    load_model()

@torch.no_grad()
def predict_image(image_np: np.ndarray, f_px: float, request_id: str, loop: asyncio.AbstractEventLoop, alpha_mask: np.ndarray = None) -> tuple:
    internal_shape = (1536, 1536)
    
    def log(msg, pct):
        if request_id in status_queues:
            loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": msg, "progress": pct}))

    # Note: Preprocessing is relatively fast, but inference is the bottleneck.
    # The lock is acquired in process_file_task before calling this function (or inside here).
    # Since prompt requested locking, we do it in the async wrapper.
    
    log("Preprocessing Image", 10)
    
    image_pt = torch.from_numpy(image_np.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    log("Running Model Inference (Heavy)", 30)
    gaussians_ndc = gaussian_predictor(image_resized_pt, disparity_factor)
    
    log("Post-processing", 70)
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )
    
    # Apply alpha mask if provided to deal with transparency
    if alpha_mask is not None:
        try:
            log("Applying Alpha Mask", 85)
            h_out = gaussian_predictor.output_resolution
            w_out = gaussian_predictor.output_resolution
            
            # Convert mask to tensor and move to device
            alpha_pt = torch.from_numpy(alpha_mask).float().to(device)
            # Add batch and channel dimensions for interpolate [1, 1, H, W]
            alpha_pt = alpha_pt[None, None]
            
            # Resize mask to output resolution
            # We use the same bilinear interpolation used for the image to stay aligned
            mask_resized = F.interpolate(
                alpha_pt,
                size=(h_out, w_out),
                mode="bilinear",
                align_corners=True,
            )
            
            # The opacities are flattened as [B, N*H*W] where N is number of layers
            # We need to determine N
            total_elements = gaussians.opacities.shape[1]
            num_layers = total_elements // (h_out * w_out)
            
            # Repeat mask for each layer and flatten to match opacities shape
            mask_final = mask_resized.repeat(1, num_layers, 1, 1).flatten(1, 3)
            
            # Multiply opacities by mask
            new_opacities = gaussians.opacities * mask_final
            gaussians = gaussians._replace(opacities=new_opacities)
            LOGGER.info(f"Applied alpha mask to {request_id}")
        except Exception as e:
            LOGGER.error(f"Failed to apply alpha mask: {e}")

    return gaussians

async def process_file_task(request_id: str, file_path: Path, output_ply_path: Path, loop: asyncio.AbstractEventLoop):
    try:
        LOGGER.info(f"Processing {file_path}")
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": "Waiting in Queue", "progress": 0}))
        
        async with inference_lock:
             # Loading image is IO, could be done outside lock, but keeping simple.
            LOGGER.info(f"Acquired lock for {request_id}")
            loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": "Loading Image", "progress": 5}))
            
            # Using run_in_executor to avoid blocking the main thread during heavy sync computations
            # However, PyTorch/MPS might need main thread sometimes on macOS? 
            # Usually heavy CPU tasks block the event loop.
            # predict_image is blocking.
            
            # For simplicity in this demo, we run synchronously in this async function, which technically blocks the loop 
            # BUT since we are inside `async with inference_lock:`, only one runs.
            # AND since we use BackgroundTasks which runs in threadpool if it was a sync def, 
            # but we made it `async def`, so it runs on main loop.
            # To avoid freezing the server (SSE heartbeats etc), we should ideally offload to thread.
            
            # Load image with potential alpha channel (remove_alpha=False)
            image_rgba, _, f_px = await asyncio.get_running_loop().run_in_executor(
                None, lambda: io.load_rgb(file_path, remove_alpha=False)
            )
            height, width = image_rgba.shape[:2]
            
            # Extract RGB and Alpha if present
            if image_rgba.shape[2] == 4:
                LOGGER.info(f"Image has alpha channel, extracting mask for {request_id}")
                image = image_rgba[:, :, :3]
                alpha_mask = image_rgba[:, :, 3] / 255.0
            else:
                image = image_rgba
                alpha_mask = None
            
            # Run prediction in executor to not block event loop (keep other SSE streams alive)
            gaussians = await asyncio.get_running_loop().run_in_executor(None, predict_image, image, f_px, request_id, loop, alpha_mask)
            
            loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": "Saving PLY File", "progress": 90}))
            LOGGER.info(f"Saving PLY to {output_ply_path}")
            await asyncio.get_running_loop().run_in_executor(None, save_ply, gaussians, f_px, (height, width), output_ply_path)

            # Skip auto-compression, return PLY URL
            # output_ksplat_path = output_ply_path.with_suffix(".ksplat")
            
            # def run_compression():
            #     try:
            #         subprocess.run(
            #             ["node", str(CONVERT_SCRIPT), str(output_ply_path.resolve()), str(output_ksplat_path.resolve())],
            #             cwd=str(FRONTEND_DIR),
            #             check=True,
            #             capture_output=True,
            #             text=True
            #         )
            #     except subprocess.CalledProcessError as e:
            #         LOGGER.error(f"Compression failed: {e.stderr}")
            #         raise Exception(f"Compression failed: {e.stderr}")

            # await asyncio.get_running_loop().run_in_executor(None, run_compression)
        
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({
            "status": "Complete", 
            "progress": 100, 
            "url": f"/outputs/{output_ply_path.name}"
        }))
    except Exception as e:
        LOGGER.error(f"Error processing {request_id}: {e}")
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": "Error", "error": str(e)}))
    finally:
        # Signal end of stream
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, None)

@app.post("/upload")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return {"error": "File must be an image"}

    request_id = str(uuid.uuid4())
    filename = file.filename or "image.png"
    if "." in filename:
        file_ext = filename.split(".")[-1]
    else:
        file_ext = "png" # Default fallback
    input_path = UPLOAD_DIR / f"{request_id}.{file_ext}"
    output_ply_path = OUTPUT_DIR / f"{request_id}.ply"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Initialize queue for this request
    status_queues[request_id] = asyncio.Queue()
    
    # Get current loop to pass to thread
    loop = asyncio.get_running_loop()
    
    # Run processing in background
    background_tasks.add_task(process_file_task, request_id, input_path, output_ply_path, loop)

    return {"status": "queued", "request_id": request_id}

@app.get("/events/{request_id}")
async def event_stream(request_id: str):
    async def event_generator():
        if request_id not in status_queues:
            yield f"data: {json.dumps({'error': 'Invalid Request ID'})}\n\n"
            return
            
        queue = status_queues[request_id]
        while True:
            data = await queue.get()
            if data is None:
                break
            yield f"data: {data}\n\n"
        
        del status_queues[request_id]

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/outputs/{filename}")
async def get_ply(filename: str):
    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        return FileResponse(file_path, media_type='application/octet-stream', filename=filename)
    return {"error": "File not found"}

@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    return {"error": "File not found"}

@app.get("/history")
async def get_history():
    generations = []
    # Find all PLY files in the outputs directory
    for ply_path in OUTPUT_DIR.glob("*.ply"):
        request_id = ply_path.stem
        
        # Find corresponding upload image
        image_url = None
        for ext in ["jpg", "jpeg", "png", "webp", "HEIC", "heic"]:
            img_path = UPLOAD_DIR / f"{request_id}.{ext}"
            if img_path.exists():
                image_url = f"/uploads/{img_path.name}"
                break
        
        # Check for other formats
        has_mesh = (OUTPUT_DIR / f"{request_id}_mesh.obj").exists()
        has_ksplat = (OUTPUT_DIR / f"{request_id}.ksplat").exists()
        
        generations.append({
            "id": request_id,
            "status": "Complete",
            "progress": 100,
            "resultUrl": f"/outputs/{ply_path.name}",
            "meshUrl": f"/outputs/{request_id}_mesh.obj" if has_mesh else None,
            "ksplatUrl": f"/outputs/{request_id}.ksplat" if has_ksplat else None,
            "imageUrl": image_url,
            "filename": ply_path.name
        })
    
    # Sort by modification time (newest first)
    generations.sort(key=lambda x: (OUTPUT_DIR / x["filename"]).stat().st_mtime, reverse=True)
    
    return generations

def mesh_conversion_worker(ply_path: Path, mesh_path: Path, log_callback):
    """Synchronous worker for heavy mesh conversion tasks."""
    try:
        import open3d as o3d
        from sharp.utils.gaussians import load_ply as load_gaussian_ply
    except ImportError:
        raise ImportError("Open3D not installed on server")

    log_callback("Loading Splat Data", 10)
    gaussians, metadata = load_gaussian_ply(ply_path)
    
    log_callback("Extracting Points", 20)
    positions = gaussians.mean_vectors.squeeze(0).cpu().numpy()
    colors = gaussians.colors.squeeze(0).cpu().numpy()
    
    # Flip orientation (Y and Z axes) to match standard 3D viewer conventions (Y-up)
    positions[:, 1] = -positions[:, 1]
    positions[:, 2] = -positions[:, 2]
    
    # Clamp colors to valid range
    colors = np.clip(colors, 0, 1)

    log_callback("Creating Point Cloud", 30)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    log_callback("Estimating Normals", 40)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    log_callback("Poisson Reconstruction", 60)
    # Poisson reconstruction returns (mesh, densities)
    # Using depth=8 to avoid "Failed to close loop" errors which occur at higher depths
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False
        )
    except Exception as e:
        LOGGER.warning(f"Poisson reconstruction failed with depth=8: {e}. Retrying with depth=6.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=6, width=0, scale=1.1, linear_fit=False
        )

    log_callback("Cleaning Mesh", 85)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    log_callback("Saving OBJ File", 95)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    
    log_callback("Complete", 100)

async def process_mesh_task(request_id: str, ply_path: Path, mesh_path: Path, mesh_filename: str, loop: asyncio.AbstractEventLoop):
    try:
        def log(msg, pct):
            if request_id in status_queues:
                loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": msg, "progress": pct}))

        # Run the heavy lifting in a separate thread so we don't block the main loop
        # This allows other requests (and SSE heartbeats) to continue processing
        await loop.run_in_executor(None, mesh_conversion_worker, ply_path, mesh_path, log)

        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({
            "status": "Complete", 
            "progress": 100, 
            "mesh_url": f"/outputs/{mesh_filename}"
        }))
    except Exception as e:
        LOGGER.error(f"Mesh conversion failed: {e}")
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": "Error", "error": str(e)}))
    finally:
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, None)

@app.post("/convert-to-mesh")
async def convert_to_mesh(background_tasks: BackgroundTasks, request: dict):
    """Convert a Gaussian splat PLY to a mesh using Poisson reconstruction."""
    ply_filename = request.get("ply_filename")
    if not ply_filename:
        return {"error": "ply_filename is required"}
    
    ply_path = OUTPUT_DIR / ply_filename
    
    # If a ksplat file is requested, try to use the original PLY file for better precision/compatibility
    if ply_path.suffix == '.ksplat':
         possible_ply = ply_path.with_suffix('.ply')
         if possible_ply.exists():
             ply_path = possible_ply
             LOGGER.info(f"Using original PLY for mesh conversion: {ply_path}")

    if not ply_path.exists():
        return {"error": "PLY file not found"}
    
    # Generate output mesh filename
    mesh_filename = ply_path.name.replace(".ply", "_mesh.obj")
    mesh_path = OUTPUT_DIR / mesh_filename
    
    request_id = str(uuid.uuid4())
    status_queues[request_id] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    background_tasks.add_task(process_mesh_task, request_id, ply_path, mesh_path, mesh_filename, loop)
    
    return {"status": "queued", "request_id": request_id}

def apply_depth_transforms_worker(ply_path: Path, output_path: Path, depth_range: tuple, depth_stretch: float, log_callback):
    """Apply depth stretch and filtering to a PLY file."""
    from sharp.utils.gaussians import load_ply as load_gaussian_ply, save_ply, Gaussians3D
    
    log_callback("Loading Splat Data", 10)
    gaussians, metadata = load_gaussian_ply(ply_path)
    
    log_callback("Applying Depth Transforms", 30)
    
    # Get mean vectors (positions)
    mean_vectors = gaussians.mean_vectors.clone()  # Shape: [1, N, 3]
    
    # The Z coordinate represents depth from origin
    z_coords = mean_vectors[0, :, 2]  # Shape: [N]
    
    # Apply depth stretch to Z coordinates
    if depth_stretch != 1.0:
        mean_vectors[0, :, 2] = z_coords * depth_stretch
        z_coords = mean_vectors[0, :, 2]  # Update after stretch
    
    log_callback("Filtering by Depth Range", 50)
    
    # Filter by depth range (based on absolute Z value from origin)
    abs_z = torch.abs(z_coords)
    min_depth, max_depth = depth_range
    mask = (abs_z >= min_depth) & (abs_z <= max_depth)
    
    # Count how many splats are kept
    original_count = mean_vectors.shape[1]
    kept_count = mask.sum().item()
    LOGGER.info(f"Depth filtering: keeping {kept_count}/{original_count} splats (range {min_depth}-{max_depth}m)")
    
    log_callback(f"Kept {kept_count}/{original_count} splats", 70)
    
    # Apply mask to all Gaussian properties
    filtered_gaussians = Gaussians3D(
        mean_vectors=mean_vectors[:, mask, :],
        quaternions=gaussians.quaternions[:, mask, :],
        singular_values=gaussians.singular_values[:, mask, :],
        opacities=gaussians.opacities[:, mask],
        colors=gaussians.colors[:, mask, :],
    )
    
    log_callback("Saving Modified PLY", 85)
    
    # Save the modified PLY
    save_ply(filtered_gaussians, metadata.focal_length_px, metadata.resolution_px, output_path)
    
    log_callback("Complete", 100)
    return kept_count, original_count

async def process_depth_transforms_task(request_id: str, ply_path: Path, output_path: Path, 
                                         depth_range: tuple, depth_stretch: float, loop: asyncio.AbstractEventLoop):
    try:
        def log(msg, pct):
            if request_id in status_queues:
                loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": msg, "progress": pct}))

        # Run the heavy lifting in a separate thread
        kept_count, original_count = await loop.run_in_executor(
            None, apply_depth_transforms_worker, ply_path, output_path, depth_range, depth_stretch, log
        )

        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({
            "status": "Complete", 
            "progress": 100, 
            "url": f"/outputs/{output_path.name}",
            "kept_count": kept_count,
            "original_count": original_count
        }))
    except Exception as e:
        LOGGER.error(f"Depth transform failed: {e}")
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": "Error", "error": str(e)}))
    finally:
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, None)

@app.post("/apply-depth-transforms")
async def apply_depth_transforms(background_tasks: BackgroundTasks, request: dict):
    """Apply depth stretch and filtering to a Gaussian splat PLY file."""
    ply_filename = request.get("ply_filename")
    depth_range = request.get("depth_range", [0.1, 100])
    depth_stretch = request.get("depth_stretch", 1.0)
    
    if not ply_filename:
        return {"error": "ply_filename is required"}
    
    ply_path = OUTPUT_DIR / ply_filename
    
    # Handle ksplat files by using original PLY
    if ply_path.suffix == '.ksplat':
        possible_ply = ply_path.with_suffix('.ply')
        if possible_ply.exists():
            ply_path = possible_ply
    
    if not ply_path.exists():
        return {"error": "PLY file not found"}
    
    # Generate output filename with transforms applied
    base_name = ply_path.stem
    output_filename = f"{base_name}_depth_modified.ply"
    output_path = OUTPUT_DIR / output_filename
    
    request_id = str(uuid.uuid4())
    status_queues[request_id] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    background_tasks.add_task(
        process_depth_transforms_task, 
        request_id, 
        ply_path, 
        output_path, 
        tuple(depth_range), 
        float(depth_stretch), 
        loop
    )
    
    return {"status": "queued", "request_id": request_id}


def filter_splats_worker(ply_path: Path, output_path: Path, brightness_threshold: float, opacity_threshold: float, log_callback):
    """Filter splats by brightness and opacity thresholds."""
    from sharp.utils.gaussians import load_ply as load_gaussian_ply, save_ply, Gaussians3D
    
    log_callback("Loading Splat Data", 10)
    gaussians, metadata = load_gaussian_ply(ply_path)
    
    log_callback("Calculating Brightness", 30)
    
    # Colors are in range [0, 1], shape: [1, N, 3]
    colors = gaussians.colors  # RGB values
    
    # Calculate perceived brightness using luminance formula
    # Y = 0.299*R + 0.587*G + 0.114*B
    brightness = 0.299 * colors[0, :, 0] + 0.587 * colors[0, :, 1] + 0.114 * colors[0, :, 2]
    
    # Opacities are in range [0, 1], shape: [1, N]
    opacities = gaussians.opacities[0, :]
    
    log_callback("Filtering Splats", 50)
    
    # Apply thresholds (thresholds are in 0-1 range)
    brightness_mask = brightness >= brightness_threshold
    opacity_mask = opacities >= opacity_threshold
    mask = brightness_mask & opacity_mask
    
    # Count results
    original_count = colors.shape[1]
    kept_count = mask.sum().item()
    removed_by_brightness = (~brightness_mask).sum().item()
    removed_by_opacity = (~opacity_mask).sum().item()
    
    LOGGER.info(f"Filter results: keeping {kept_count}/{original_count} splats")
    LOGGER.info(f"  - Removed by brightness < {brightness_threshold:.3f}: {removed_by_brightness}")
    LOGGER.info(f"  - Removed by opacity < {opacity_threshold:.3f}: {removed_by_opacity}")
    
    log_callback(f"Kept {kept_count}/{original_count} splats", 70)
    
    # Apply mask to all Gaussian properties
    filtered_gaussians = Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask, :],
        quaternions=gaussians.quaternions[:, mask, :],
        singular_values=gaussians.singular_values[:, mask, :],
        opacities=gaussians.opacities[:, mask],
        colors=gaussians.colors[:, mask, :],
    )
    
    log_callback("Saving Filtered PLY", 85)
    
    # Save the filtered PLY
    save_ply(filtered_gaussians, metadata.focal_length_px, metadata.resolution_px, output_path)
    
    log_callback("Complete", 100)
    return kept_count, original_count


async def process_filter_task(request_id: str, ply_path: Path, output_path: Path, 
                               brightness_threshold: float, opacity_threshold: float, loop: asyncio.AbstractEventLoop):
    try:
        def log(msg, pct):
            if request_id in status_queues:
                loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": msg, "progress": pct}))

        # Run the heavy lifting in a separate thread
        kept_count, original_count = await loop.run_in_executor(
            None, filter_splats_worker, ply_path, output_path, brightness_threshold, opacity_threshold, log
        )

        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({
            "status": "Complete", 
            "progress": 100, 
            "url": f"/outputs/{output_path.name}",
            "kept_count": kept_count,
            "original_count": original_count
        }))
    except Exception as e:
        LOGGER.error(f"Filter failed: {e}")
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": "Error", "error": str(e)}))
    finally:
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, None)


@app.post("/filter-splats")
async def filter_splats(background_tasks: BackgroundTasks, request: dict):
    """Filter splats by brightness and opacity thresholds."""
    ply_filename = request.get("ply_filename")
    brightness_threshold = request.get("brightness_threshold", 0.0)  # 0-1 range
    opacity_threshold = request.get("opacity_threshold", 0.0)  # 0-1 range
    
    if not ply_filename:
        return {"error": "ply_filename is required"}
    
    ply_path = OUTPUT_DIR / ply_filename
    
    # Handle ksplat files by using original PLY
    if ply_path.suffix == '.ksplat':
        possible_ply = ply_path.with_suffix('.ply')
        if possible_ply.exists():
            ply_path = possible_ply
    
    if not ply_path.exists():
        return {"error": "PLY file not found"}
    
    # Generate output filename
    base_name = ply_path.stem
    # Remove any previous filter suffix
    if "_filtered" in base_name:
        base_name = base_name.split("_filtered")[0]
    output_filename = f"{base_name}_filtered.ply"
    output_path = OUTPUT_DIR / output_filename
    
    request_id = str(uuid.uuid4())
    status_queues[request_id] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    background_tasks.add_task(
        process_filter_task, 
        request_id, 
        ply_path, 
        output_path, 
        float(brightness_threshold),
        float(opacity_threshold),
        loop
    )
    
    return {"status": "queued", "request_id": request_id}


@app.post("/convert-format")
async def convert_format(background_tasks: BackgroundTasks, request: dict):
    """Convert a PLY file to another format (SPZ, SOG, etc.) using 3dgsconverter."""
    ply_filename = request.get("ply_filename")
    target_format = request.get("format")
    options = request.get("options", {})
    
    if not ply_filename:
        return {"error": "ply_filename is required"}
    if not target_format:
        return {"error": "format is required"}
        
    ply_path = OUTPUT_DIR / ply_filename
    
    # Handle ksplat input if passed (use original ply)
    if ply_path.suffix == '.ksplat':
         possible_ply = ply_path.with_suffix('.ply')
         if possible_ply.exists():
             ply_path = possible_ply
    
    if not ply_path.exists():
        return {"error": "PLY file not found"}
        
    # Generate output filename
    # Remove extension and add new one
    output_filename = f"{ply_path.stem}.{target_format}"
    output_path = OUTPUT_DIR / output_filename
    
    request_id = str(uuid.uuid4())
    status_queues[request_id] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    background_tasks.add_task(
        process_conversion_task, 
        request_id, 
        ply_path, 
        output_path, 
        target_format,
        options,
        loop
    )
    
    return {"status": "queued", "request_id": request_id}

async def process_conversion_task(request_id: str, ply_path: Path, output_path: Path, 
                                   target_format: str, options: dict, loop: asyncio.AbstractEventLoop):
    try:
        def log(msg, pct):
            if request_id in status_queues:
                loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": msg, "progress": pct}))

        log(f"Starting {target_format.upper()} Conversion", 5)
        
        # Build command
        # Find 3dgsconverter in the same directory as python executable (venv/bin)
        python_bin_dir = Path(sys.executable).parent
        converter_exec = "3dgsconverter" # default fallback to path
        
        possible_paths = [
            python_bin_dir / "3dgsconverter",
            python_bin_dir / "3dgsconverter.exe"
        ]
        
        for p in possible_paths:
            if p.exists():
                converter_exec = str(p)
                break
        
        cmd = [converter_exec, "--input", str(ply_path), "--output", str(output_path), "--force"]
        
        # Add options
        if "compression_level" in options:
             cmd.extend(["--compression_level", str(options["compression_level"])])
        
        # If the tool supports target format explicitly (it seems it infers from output, but let's check)
        # help said: [--target_format TARGET_FORMAT]
        cmd.extend(["--target_format", target_format])

        log("Running Converter Tool", 20)
        LOGGER.info(f"Running conversion command: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for process
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode()
            LOGGER.error(f"Conversion failed: {error_msg}")
            raise Exception(f"Conversion tool failed: {error_msg}")
            
        log("Conversion Complete", 100)
        
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({
            "status": "Complete", 
            "progress": 100, 
            "url": f"/outputs/{output_path.name}"
        }))
        
    except Exception as e:
        LOGGER.error(f"Conversion failed: {e}")
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": "Error", "error": str(e)}))
    finally:
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, None)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

