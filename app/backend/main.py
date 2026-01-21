
import logging
import shutil
import uuid
import asyncio
import json
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
def predict_image(image_np: np.ndarray, f_px: float, request_id: str, loop: asyncio.AbstractEventLoop) -> tuple:
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
            
            image, _, f_px = await asyncio.get_running_loop().run_in_executor(None, io.load_rgb, file_path)
            height, width = image.shape[:2]
            
            # Run prediction in executor to not block event loop (keep other SSE streams alive)
            gaussians = await asyncio.get_running_loop().run_in_executor(None, predict_image, image, f_px, request_id, loop)
            
            loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": "Saving PLY File", "progress": 90}))
            LOGGER.info(f"Saving PLY to {output_ply_path}")
            await asyncio.get_running_loop().run_in_executor(None, save_ply, gaussians, f_px, (height, width), output_ply_path)

            loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({"status": "Compressing", "progress": 95}))
            LOGGER.info(f"Compressing {output_ply_path} to ksplat...")
            
            output_ksplat_path = output_ply_path.with_suffix(".ksplat")
            
            def run_compression():
                try:
                    subprocess.run(
                        ["node", str(CONVERT_SCRIPT), str(output_ply_path.resolve()), str(output_ksplat_path.resolve())],
                        cwd=str(FRONTEND_DIR),
                        check=True,
                        capture_output=True,
                        text=True
                    )
                except subprocess.CalledProcessError as e:
                    LOGGER.error(f"Compression failed: {e.stderr}")
                    raise Exception(f"Compression failed: {e.stderr}")

            await asyncio.get_running_loop().run_in_executor(None, run_compression)
        
        loop.call_soon_threadsafe(status_queues[request_id].put_nowait, json.dumps({
            "status": "Complete", 
            "progress": 100, 
            "url": f"/outputs/{output_ksplat_path.name}"
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

@app.post("/convert-to-mesh")
async def convert_to_mesh(request: dict):
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
    
    try:
        import open3d as o3d
        
        LOGGER.info(f"Converting {ply_path} to mesh...")
        
        # Load the Gaussian splat PLY file
        # We need to manually parse it since it's a Gaussian splat format, not standard point cloud
        from sharp.utils.gaussians import load_ply as load_gaussian_ply
        
        gaussians, metadata = load_gaussian_ply(ply_path)
        
        # Extract point positions from Gaussians
        positions = gaussians.mean_vectors.squeeze(0).cpu().numpy()
        colors = gaussians.colors.squeeze(0).cpu().numpy()
        
        # Clamp colors to valid range
        colors = np.clip(colors, 0, 1)
        
        LOGGER.info(f"Extracted {len(positions)} points from Gaussian splat")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals for Poisson reconstruction
        LOGGER.info("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # Perform Poisson surface reconstruction
        LOGGER.info("Performing Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, width=0, scale=1.1, linear_fit=False
        )
        
        # Remove low-density vertices (noise)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.05)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Transfer colors from point cloud to mesh vertices
        mesh.compute_vertex_normals()
        
        # Save the mesh
        LOGGER.info(f"Saving mesh to {mesh_path}")
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        
        return {"mesh_url": f"/outputs/{mesh_filename}"}
        
    except ImportError:
        LOGGER.error("Open3D not installed. Please install it with: pip install open3d")
        return {"error": "Open3D is required for mesh conversion. Install with: pip install open3d"}
    except Exception as e:
        LOGGER.error(f"Mesh conversion failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

