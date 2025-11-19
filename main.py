from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa
from skimage.transform import resize
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

app = FastAPI(title="String Art API - Detailed Focus", description="High-detail string art generation")

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

def detailed_preprocessing_v1(img: np.ndarray) -> np.ndarray:
    """Shade 1: Strong detail with deep shadows"""
    # Light smoothing
    smoothed = gaussian_filter(img, sigma=0.5)
    
    # Moderate contrast boost
    mean = np.mean(smoothed)
    contrasted = (smoothed - mean) * 1.25 + mean
    contrasted = np.clip(contrasted, 0, 1)
    
    # Darken for deep shadows (makes strings pop more)
    result = contrasted * 0.85
    
    return np.clip(result, 0, 1)

def detailed_preprocessing_v2(img: np.ndarray) -> np.ndarray:
    """Shade 2: Balanced detail with medium contrast"""
    # Light smoothing
    smoothed = gaussian_filter(img, sigma=0.5)
    
    # Balanced contrast
    mean = np.mean(smoothed)
    contrasted = (smoothed - mean) * 1.20 + mean
    contrasted = np.clip(contrasted, 0, 1)
    
    # Medium darkening
    result = contrasted * 0.88
    
    return np.clip(result, 0, 1)

def detailed_preprocessing_v3(img: np.ndarray) -> np.ndarray:
    """Shade 3: Softer detail with lighter tones"""
    # Slightly more smoothing for softer look
    smoothed = gaussian_filter(img, sigma=0.6)
    
    # Gentle contrast
    mean = np.mean(smoothed)
    contrasted = (smoothed - mean) * 1.18 + mean
    contrasted = np.clip(contrasted, 0, 1)
    
    # Light darkening (brighter overall)
    result = contrasted * 0.90
    
    return np.clip(result, 0, 1)

# ============== NAIL POSITIONING ==============

def create_circle_nails(shape, nail_count=200):
    """Create nail positions with proper boundary checking"""
    h, w = shape
    cy, cx = h // 2, w // 2
    # Use 95% of radius to ensure nails stay inside
    radius = int(min(h, w) * 0.475)
    
    nails = []
    for i in range(nail_count):
        angle = 2 * np.pi * i / nail_count
        y = int(cy + radius * np.sin(angle))
        x = int(cx + radius * np.cos(angle))
        # Clamp to valid range
        y = max(0, min(h - 1, y))
        x = max(0, min(w - 1, x))
        nails.append((y, x))
    
    return np.array(nails, dtype=np.int32)

# ============== CORE ALGORITHM ==============

def draw_line_safe(canvas, p0, p1, strength):
    """Draw line with boundary checking"""
    h, w = canvas.shape
    rr, cc, val = line_aa(p0[0], p0[1], p1[0], p1[1])
    
    # Keep only valid pixels
    mask = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rr, cc, val = rr[mask], cc[mask], val[mask]
    
    if len(rr) == 0:
        return None
    
    canvas[rr, cc] = np.clip(canvas[rr, cc] + val * strength, 0.0, 1.0)
    return rr, cc, val

def calc_improvement(canvas, target, p0, p1, strength):
    """Calculate how much a line would improve the image"""
    h, w = canvas.shape
    rr, cc, val = line_aa(p0[0], p0[1], p1[0], p1[1])
    
    # Boundary check
    mask = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rr, cc, val = rr[mask], cc[mask], val[mask]
    
    if len(rr) == 0:
        return -1.0, None
    
    # Calculate improvement in squared error
    before = (canvas[rr, cc] - target[rr, cc]) ** 2
    after_pixels = np.clip(canvas[rr, cc] + val * strength, 0.0, 1.0)
    after = (after_pixels - target[rr, cc]) ** 2
    
    improvement = np.sum(before - after)
    return float(improvement), (rr, cc, val)

def greedy_string_art(nails, target, max_iterations=4000, strength=-0.030, 
                     random_sample=80, fail_limit=10):
    """
    HIGH DETAIL greedy algorithm with more iterations
    """
    h, w = target.shape
    canvas = np.ones((h, w), dtype=np.float32)
    
    current_idx = 0
    order = [0]
    fails = 0
    n_nails = len(nails)
    
    start_time = time()
    iter_times = []
    
    for iteration in range(max_iterations):
        iter_start = time()
        
        if iteration % 500 == 0:
            logger.info(f"  Iteration {iteration}, Pulls: {len(order)}, Fails: {fails}")
        
        # Sample random candidates
        candidates = np.random.choice(
            [i for i in range(n_nails) if i != current_idx],
            size=min(random_sample, n_nails - 1),
            replace=False
        )
        
        best_improvement = -1e9
        best_idx = None
        best_data = None
        
        # Find best candidate
        for cand_idx in candidates:
            imp, data = calc_improvement(
                canvas, target, nails[current_idx], nails[cand_idx], strength
            )
            if imp > best_improvement:
                best_improvement = imp
                best_idx = cand_idx
                best_data = data
        
        # Apply best move or escape
        if best_improvement <= 0:
            fails += 1
            if fails >= fail_limit:
                break
            # Escape: jump to a random far nail
            far_candidates = [i for i in range(n_nails) 
                            if abs(i - current_idx) > n_nails // 8]
            if far_candidates:
                jump_idx = np.random.choice(far_candidates)
                draw_line_safe(canvas, nails[current_idx], nails[jump_idx], strength)
                current_idx = jump_idx
                order.append(current_idx)
        else:
            # Apply the good move
            fails = 0
            rr, cc, val = best_data
            canvas[rr, cc] = np.clip(canvas[rr, cc] + val * strength, 0.0, 1.0)
            current_idx = best_idx
            order.append(current_idx)
        
        iter_times.append(time() - iter_start)
    
    elapsed = time() - start_time
    logger.info(f"  Completed: {len(order)} pulls in {elapsed:.1f}s (avg {np.mean(iter_times):.4f}s/iter)")
    
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
        logger.info(f"[{job_id}] Starting HIGH DETAIL job")
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
        work_size = int(params.get("work_size", 300))
        gray_resized = resize(gray, (work_size, work_size), anti_aliasing=True)
        
        # Create nails
        nail_count = int(params.get("nail_count", 200))
        nails = create_circle_nails(gray_resized.shape, nail_count)
        labels = create_nail_labels(nail_count)
        
        logger.info(f"[{job_id}] Image: {work_size}x{work_size}, Nails: {nail_count}")
        
        # Process 3 DETAILED variants with different shades
        variants = {
            'detailed_deep': {
                'preprocess': detailed_preprocessing_v1,
                'strength': -0.032,
                'iters': 4000,
                'desc': 'Deep shadows - strongest detail (DARKEST)'
            },
            'detailed_balanced': {
                'preprocess': detailed_preprocessing_v2,
                'strength': -0.030,
                'iters': 4200,
                'desc': 'Balanced detail - recommended (MEDIUM)'
            },
            'detailed_bright': {
                'preprocess': detailed_preprocessing_v3,
                'strength': -0.028,
                'iters': 4500,
                'desc': 'Lighter tones - softer detail (BRIGHTEST)'
            }
        }
        
        results = {}
        
        for i, (name, config) in enumerate(variants.items(), 1):
            logger.info(f"[{job_id}] Processing DETAILED variant {i}/3: {name}")
            
            # Preprocess
            processed = config['preprocess'](gray_resized)
            
            # Run algorithm with MORE ITERATIONS for detail
            canvas, order = greedy_string_art(
                nails, processed,
                max_iterations=config['iters'],
                strength=config['strength'],
                random_sample=params.get("random_sample", 80),
                fail_limit=params.get("fail_limit", 10)
            )
            
            # Render at export size
            export_size = int(params.get("export_size", 600))
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
            
            del canvas, processed
            gc.collect()
        
        # Complete job
        jobs_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "completed_at": datetime.now().isoformat(),
            "results": results,
            "metadata": {
                "nails_count": nail_count,
                "work_size": work_size,
                "export_size": params.get("export_size", 600),
                "nail_labels": labels,
                "variants": 3,
                "focus": "HIGH DETAIL ONLY - 3 shades"
            }
        })
        
        logger.info(f"[{job_id}] SUCCESS - 3 DETAILED variants completed")
        
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
        "message": "String Art API - HIGH DETAIL FOCUS",
        "status": "healthy",
        "variants": 3,
        "focus": "Detailed only - 3 different shades",
        "features": [
            "4000-4500 iterations per variant",
            "Deep shadows, balanced, and bright versions",
            "Maximum detail and clarity",
            "5-7 minute processing time"
        ]
    }

@app.post("/jobs")
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    work_size: int = 300,
    export_size: int = 600,
    nail_count: int = 200,
    random_sample: int = 80,
    fail_limit: int = 10
):
    """
    Submit HIGH DETAIL string art job - returns 3 detailed variants:
    
    1. Detailed Deep - 4000 iterations, darkest, strongest contrast
    2. Detailed Balanced - 4200 iterations, medium tones (RECOMMENDED)
    3. Detailed Bright - 4500 iterations, lighter, softer
    
    Processing time: 5-7 minutes total
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    job_id = str(uuid.uuid4())
    image_data = await file.read()
    
    params = {
        "work_size": work_size,
        "export_size": export_size,
        "nail_count": nail_count,
        "random_sample": random_sample,
        "fail_limit": fail_limit
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
        "message": "Job queued - will generate 3 DETAILED variants",
        "estimated_time": "5-7 minutes",
        "variants": 3,
        "detail_level": "HIGH - focused on maximum quality"
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