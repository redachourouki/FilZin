from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa
from skimage.transform import resize
from skimage import filters, exposure
from scipy.ndimage import gaussian_filter
import cv2
from time import time
from io import BytesIO
from PIL import Image
import uuid
from typing import Dict, Any, Optional
from enum import Enum
import gc
import logging
from datetime import datetime
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="String Art API - WowStrings Enhanced", description="High-quality string art with advanced algorithms")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

jobs_store: Dict[str, Dict] = {}

# ============== NAIL LABELING ==============

def create_nail_labels(nail_count=200):
    labels = []
    sections = ['A', 'B', 'C', 'D']
    per_section = nail_count // 4
    for section in sections:
        for i in range(1, per_section + 1):
            labels.append(f"{section}{i}")
    return labels

def pull_order_to_labels(order, labels):
    return [labels[i] if 0 <= i < len(labels) else str(i) for i in order]

# ============== IMAGE PROCESSING ==============

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def largest_center_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    size = min(h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return img[top:top + size, left:left + size]

def wowstrings_preprocessing(img: np.ndarray, variant='balanced') -> tuple:
    """
    WowStrings-style preprocessing with edge detection and adaptive contrast
    Returns: (processed_image, edge_map, importance_map)
    """
    # Step 1: Gentle smoothing to reduce noise
    smoothed = gaussian_filter(img, sigma=0.8)
    
    # Step 2: Detect edges (important features)
    edges = filters.sobel(smoothed)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    
    # Step 3: Create importance map (edges + dark areas)
    importance = edges * 0.6 + (1 - smoothed) * 0.4
    
    # Step 4: Adaptive histogram equalization for local contrast
    clahe_img = exposure.equalize_adapthist(smoothed, clip_limit=0.03)
    
    # Step 5: Apply variant-specific processing
    if variant == 'extreme':
        # Very dark and high contrast
        contrast_boost = 1.5
        darkness = 0.65
    elif variant == 'dark':
        # Dark with good detail
        contrast_boost = 1.4
        darkness = 0.70
    else:  # balanced
        # Balanced approach
        contrast_boost = 1.3
        darkness = 0.75
    
    # Boost contrast
    mean = np.mean(clahe_img)
    contrasted = (clahe_img - mean) * contrast_boost + mean
    contrasted = np.clip(contrasted, 0, 1)
    
    # Apply darkness
    processed = contrasted * darkness
    
    # Step 6: Sharpen important features
    sharpened = processed - 0.2 * gaussian_filter(processed, sigma=2)
    result = np.clip(sharpened, 0, 1)
    
    return result, edges, importance

# ============== NAIL POSITIONING ==============

def create_circle_nails(shape, nail_count=200):
    """Create nail positions with proper boundary checking"""
    h, w = shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * 0.475)
    
    nails = []
    for i in range(nail_count):
        angle = 2 * np.pi * i / nail_count
        y = int(cy + radius * np.sin(angle))
        x = int(cx + radius * np.cos(angle))
        y = max(0, min(h - 1, y))
        x = max(0, min(w - 1, x))
        nails.append((y, x))
    
    return np.array(nails, dtype=np.int32)

# ============== CORE ALGORITHM ==============

def draw_line_safe(canvas, p0, p1, strength):
    """Draw line with boundary checking"""
    h, w = canvas.shape
    rr, cc, val = line_aa(p0[0], p0[1], p1[0], p1[1])
    
    mask = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rr, cc, val = rr[mask], cc[mask], val[mask]
    
    if len(rr) == 0:
        return None
    
    canvas[rr, cc] = np.clip(canvas[rr, cc] + val * strength, 0.0, 1.0)
    return rr, cc, val

def calc_improvement_weighted(canvas, target, importance_map, p0, p1, strength):
    """
    Calculate improvement with importance weighting
    This prioritizes lines that cross important features (edges, dark areas)
    """
    h, w = canvas.shape
    rr, cc, val = line_aa(p0[0], p0[1], p1[0], p1[1])
    
    mask = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rr, cc, val = rr[mask], cc[mask], val[mask]
    
    if len(rr) == 0:
        return -1.0, None
    
    # Calculate error reduction
    before = (canvas[rr, cc] - target[rr, cc]) ** 2
    after_pixels = np.clip(canvas[rr, cc] + val * strength, 0.0, 1.0)
    after = (after_pixels - target[rr, cc]) ** 2
    
    # Weight by importance (prioritize edges and features)
    weights = importance_map[rr, cc] * 2.0 + 0.5  # Boost important areas
    improvement = np.sum((before - after) * weights)
    
    return float(improvement), (rr, cc, val)

def wowstrings_algorithm(nails, target, importance_map, target_strings=4000, 
                         initial_strength=-0.042, min_strength=-0.025,
                         random_sample=120, fail_limit=25):
    """
    WowStrings-style algorithm with EXACT string count control:
    - Adaptive line strength (starts strong, gets weaker for detail)
    - Importance-weighted line selection
    - Runs until exactly target_strings is reached
    - More patience for finding good lines
    """
    h, w = target.shape
    canvas = np.ones((h, w), dtype=np.float32)
    
    current_idx = 0
    order = [0]
    fails = 0
    n_nails = len(nails)
    
    start_time = time()
    max_iterations = target_strings + 1000  # Safety buffer
    
    for iteration in range(max_iterations):
        # Stop when we reach target string count
        if len(order) >= target_strings:
            logger.info(f"  Target reached: {len(order)} strings")
            break
            
        if iteration % 500 == 0:
            logger.info(f"  Progress: {len(order)}/{target_strings} strings, Fails: {fails}")
        
        # Adaptive strength: start strong for coverage, get weaker for detail
        progress = len(order) / target_strings
        strength = initial_strength + (min_strength - initial_strength) * progress
        
        # Sample candidates (avoid recent nails)
        recent_nails = set(order[-10:]) if len(order) > 10 else set()
        available = [i for i in range(n_nails) if i != current_idx and i not in recent_nails]
        
        if len(available) == 0:
            available = [i for i in range(n_nails) if i != current_idx]
        
        candidates = np.random.choice(
            available,
            size=min(random_sample, len(available)),
            replace=False
        )
        
        best_improvement = -1e9
        best_idx = None
        best_data = None
        
        # Find best line with importance weighting
        for cand_idx in candidates:
            imp, data = calc_improvement_weighted(
                canvas, target, importance_map, 
                nails[current_idx], nails[cand_idx], strength
            )
            if imp > best_improvement:
                best_improvement = imp
                best_idx = cand_idx
                best_data = data
        
        if best_improvement <= 0:
            fails += 1
            if fails >= fail_limit:
                # Reset fails and force a jump
                fails = 0
            
            # Jump to distant nail
            far_candidates = [i for i in range(n_nails) 
                            if abs(i - current_idx) > n_nails // 6]
            if far_candidates:
                jump_idx = np.random.choice(far_candidates)
                draw_line_safe(canvas, nails[current_idx], nails[jump_idx], strength)
                current_idx = jump_idx
                order.append(current_idx)
        else:
            fails = 0
            rr, cc, val = best_data
            canvas[rr, cc] = np.clip(canvas[rr, cc] + val * strength, 0.0, 1.0)
            current_idx = best_idx
            order.append(current_idx)
    
    elapsed = time() - start_time
    logger.info(f"  Completed: {len(order)} strings in {elapsed:.1f}s")
    
    return canvas, order

# ============== RENDERING ==============

def array_to_base64(img_array: np.ndarray) -> str:
    """Convert array to base64 PNG"""
    fig = plt.figure(figsize=(8, 8), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img_array, cmap='gray', vmin=0.0, vmax=1.0)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close(fig)
    
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    gc.collect()
    
    return b64

# ============== JOB PROCESSING ==============

def process_job(job_id: str, image_data: bytes, params: Dict[str, Any]):
    try:
        logger.info(f"[{job_id}] Starting WowStrings-Enhanced job")
        jobs_store[job_id]["status"] = JobStatus.PROCESSING
        jobs_store[job_id]["started_at"] = datetime.now().isoformat()
        
        # Load image
        pil = Image.open(BytesIO(image_data)).convert("RGB")
        img = np.array(pil, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        # Preprocessing
        gray = rgb2gray(img)
        gray = largest_center_square(gray)
        
        # Resize to working size
        work_size = int(params.get("work_size", 400))
        gray_resized = resize(gray, (work_size, work_size), anti_aliasing=True)
        
        # Create nails
        nail_count = int(params.get("nail_count", 200))
        nails = create_circle_nails(gray_resized.shape, nail_count)
        labels = create_nail_labels(nail_count)
        
        logger.info(f"[{job_id}] Image: {work_size}x{work_size}, Nails: {nail_count}")
        
        # Process 3 variants with exact string counts
        variants = {
            '3500_strings': {
                'variant': 'balanced',
                'target_strings': 3500,
                'initial_strength': -0.038,
                'min_strength': -0.024,
                'desc': '3,500 strings - Light and detailed'
            },
            '4000_strings': {
                'variant': 'dark',
                'target_strings': 4000,
                'initial_strength': -0.041,
                'min_strength': -0.026,
                'desc': '4,000 strings - Balanced density (RECOMMENDED)'
            },
            '4500_strings': {
                'variant': 'extreme',
                'target_strings': 4500,
                'initial_strength': -0.044,
                'min_strength': -0.028,
                'desc': '4,500 strings - Maximum density and detail'
            }
        }
        
        results = {}
        
        for i, (name, config) in enumerate(variants.items(), 1):
            logger.info(f"[{job_id}] Processing variant {i}/3: {name}")
            
            # Advanced preprocessing with edge detection
            processed, edges, importance = wowstrings_preprocessing(
                gray_resized, 
                variant=config['variant']
            )
            
            # Run enhanced algorithm with exact string count
            canvas, order = wowstrings_algorithm(
                nails, processed, importance,
                target_strings=config['target_strings'],
                initial_strength=config['initial_strength'],
                min_strength=config['min_strength'],
                random_sample=120,
                fail_limit=25
            )
            
            # Render at export size
            export_size = int(params.get("export_size", 800))
            if export_size != work_size:
                canvas_resized = resize(canvas, (export_size, export_size), anti_aliasing=True)
            else:
                canvas_resized = canvas
            
            # Convert to base64
            b64 = array_to_base64(canvas_resized)
            
            # Create labeled order
            order_labeled = pull_order_to_labels(order, labels)
            
            results[name] = {
                "image_base64": b64,
                "pull_order": "-".join(order_labeled),
                "pull_order_numeric": "-".join(str(x) for x in order),
                "total_pulls": len(order),
                "description": config['desc'],
                "variant_number": i
            }
            
            del canvas, processed, edges, importance
            gc.collect()
        
        # Complete job
        jobs_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "completed_at": datetime.now().isoformat(),
            "results": results,
            "metadata": {
                "nails_count": nail_count,
                "work_size": work_size,
                "export_size": params.get("export_size", 800),
                "nail_labels": labels,
                "variants": 3,
                "quality": "WowStrings-Enhanced - Exact string counts with edge detection"
            }
        })
        
        logger.info(f"[{job_id}] SUCCESS - 3 WowStrings-Enhanced variants completed")
        
    except Exception as e:
        logger.exception(f"[{job_id}] FAILED: {e}")
        jobs_store[job_id].update({
            "status": JobStatus.FAILED,
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
    finally:
        gc.collect()

# ============== API ENDPOINTS ==============

@app.get("/")
async def root():
    return {
        "message": "String Art API - WowStrings Enhanced",
        "status": "healthy",
        "variants": 3,
        "quality": "Advanced algorithm with edge detection",
        "features": [
            "Exact string counts: 3500, 4000, 4500",
            "Fixed 200 nails",
            "Adaptive line strength (strong → weak for detail)",
            "Importance-weighted line selection (prioritizes edges)",
            "Advanced preprocessing with CLAHE",
            "Edge detection for feature emphasis",
            "Smarter candidate sampling"
        ],
        "processing_time": "6-10 minutes for all 3 variants"
    }

@app.post("/jobs")
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    work_size: int = 400,
    export_size: int = 800,
    nail_count: int = 200
):
    """
    Submit WowStrings-Enhanced job - returns 3 variants:
    
    1. 3,500 strings - Light and detailed
    2. 4,000 strings - Balanced density (RECOMMENDED)
    3. 4,500 strings - Maximum density
    
    Fixed: 200 nails
    
    Features:
    - Exact string count control
    - Edge detection for feature emphasis
    - Adaptive line strength for better detail
    - Importance-weighted line selection
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    job_id = str(uuid.uuid4())
    image_data = await file.read()
    
    params = {
        "work_size": work_size,
        "export_size": export_size,
        "nail_count": nail_count
    }
    
    jobs_store[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        "filename": file.filename,
        "params": params
    }
    
    background_tasks.add_task(process_job, job_id, image_data, params)
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Job queued - WowStrings Enhanced Mode",
        "estimated_time": "6-10 minutes",
        "variants": 3,
        "string_counts": [3500, 4000, 4500],
        "nail_count": 200,
        "quality_level": "Exact string counts with edge detection & adaptive strength"
    }

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs_store:
        raise HTTPException(404, "Job not found")
    return jobs_store[job_id]

@app.get("/health")
async def health():
    return {"status": "healthy", "active_jobs": len(jobs_store)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")