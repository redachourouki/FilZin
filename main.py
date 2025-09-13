from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.draw import line_aa, ellipse_perimeter
from math import atan2
from skimage.transform import resize
from time import time
import tempfile
import os
import base64
from io import BytesIO
from PIL import Image
import uuid
from typing import Dict, Any, Optional
from enum import Enum
import gc
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="String Art API", description="Generate string art from images with async job processing")

# Enable CORS for Bubble integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job status enum
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

# In-memory job storage (use Redis/DB in production)
jobs_store: Dict[str, Dict] = {}

# ============== NAIL LABELING SYSTEM ==============

def create_nail_labels(nail_count=200):
    """Create sectioned nail labels: A1-A50, B1-B50, C1-C50, D1-D50"""
    labels = []
    sections = ['A', 'B', 'C', 'D']
    nails_per_section = nail_count // 4  # 50 nails per section
    
    for section_idx, section in enumerate(sections):
        for nail_num in range(1, nails_per_section + 1):
            labels.append(f"{section}{nail_num}")
    
    return labels

def index_to_label(index, nail_labels):
    """Convert numeric index to nail label"""
    if 0 <= index < len(nail_labels):
        return nail_labels[index]
    return str(index)  # fallback to number if out of range

def pull_order_to_labels(pull_order, nail_labels):
    """Convert numeric pull order to labeled pull order"""
    return [index_to_label(idx, nail_labels) for idx in pull_order]

# ============== ORIGINAL ALGORITHM CODE (UPDATED) ==============

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def adjust_contrast(image, contrast_factor):
    adjusted = (image - 0.5) * contrast_factor + 0.5
    return np.clip(adjusted, 0, 1)

def adjust_brightness(image, brightness_factor):
    adjusted = image + brightness_factor
    return np.clip(adjusted, 0, 1)

def largest_square(image: np.ndarray) -> np.ndarray:
    short_edge = np.argmin(image.shape[:2])
    short_edge_half = image.shape[short_edge] // 2
    long_edge_center = image.shape[1 - short_edge] // 2
    if short_edge == 0:
        return image[:, long_edge_center - short_edge_half:
                        long_edge_center + short_edge_half]
    if short_edge == 1:
        return image[long_edge_center - short_edge_half:
                     long_edge_center + short_edge_half, :]

def create_rectangle_nail_positions(shape, nail_step=2):
    height, width = shape
    nails_top = [(0, i) for i in range(0, width, nail_step)]
    nails_bot = [(height-1, i) for i in range(0, width, nail_step)]
    nails_right = [(i, width-1) for i in range(1, height-1, nail_step)]
    nails_left = [(i, 0) for i in range(1, height-1, nail_step)]
    nails = nails_top + nails_right + nails_bot + nails_left
    return np.array(nails)

def create_circle_nail_positions(shape, nail_count=200, r1_multip=1, r2_multip=1):
    height = shape[0]
    width = shape[1]
    centre = (height // 2, width // 2)
    radius = min(height, width) // 2 - 1
    
    nails = []
    for i in range(nail_count):
        angle = 2 * np.pi * i / nail_count
        y = int(centre[0] + radius * r1_multip * np.sin(angle))
        x = int(centre[1] + radius * r2_multip * np.cos(angle))
        nails.append((y, x))
    
    return np.asarray(nails)

def init_canvas(shape, black=False):
    if black:
        return np.zeros(shape)
    else:
        return np.ones(shape)

def get_aa_line(from_pos, to_pos, str_strength, picture):
    rr, cc, val = line_aa(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
    line = picture[rr, cc] + str_strength * val
    line = np.clip(line, a_min=0, a_max=1)
    return line, rr, cc

def find_best_nail_position(current_position, nails, str_pic, orig_pic, str_strength, random_nails=None):
    best_cumulative_improvement = -99999
    best_nail_position = None
    best_nail_idx = None
    
    if random_nails is not None:
        nail_ids = np.random.choice(range(len(nails)), size=random_nails, replace=False)
        nails_and_ids = list(zip(nail_ids, nails[nail_ids]))
    else:
        nails_and_ids = enumerate(nails)

    for nail_idx, nail_position in nails_and_ids:
        overlayed_line, rr, cc = get_aa_line(current_position, nail_position, str_strength, str_pic)
        before_overlayed_line_diff = np.abs(str_pic[rr, cc] - orig_pic[rr, cc])**2
        after_overlayed_line_diff = np.abs(overlayed_line - orig_pic[rr, cc])**2
        cumulative_improvement = np.sum(before_overlayed_line_diff - after_overlayed_line_diff)

        if cumulative_improvement >= best_cumulative_improvement:
            best_cumulative_improvement = cumulative_improvement
            best_nail_position = nail_position
            best_nail_idx = nail_idx

    return best_nail_idx, best_nail_position, best_cumulative_improvement

def create_art(nails, orig_pic, str_pic, str_strength, i_limit=None, random_nails=None):
    start = time()
    iter_times = []
    current_position = nails[0]
    pull_order = [0]
    i = 0
    fails = 0
    
    while True:
        start_iter = time()
        i += 1
        
        if i%500 == 0:
            logger.info(f"Iteration {i}")
        
        if i_limit == None:
            if fails >= 3:
                break
        else:
            if i > i_limit:
                break

        idx, best_nail_position, best_cumulative_improvement = find_best_nail_position(
            current_position, nails, str_pic, orig_pic, str_strength, random_nails
        )

        if best_cumulative_improvement <= 0:
            fails += 1
            continue

        pull_order.append(idx)
        best_overlayed_line, rr, cc = get_aa_line(current_position, best_nail_position, str_strength, str_pic)
        str_pic[rr, cc] = best_overlayed_line
        current_position = best_nail_position
        iter_times.append(time() - start_iter)

    logger.info(f"Time: {time() - start}")
    logger.info(f"Avg iteration time: {np.mean(iter_times) if iter_times else 0}")
    return pull_order

def scale_nails(x_ratio, y_ratio, nails):
    return [(int(y_ratio*nail[0]), int(x_ratio*nail[1])) for nail in nails]

def pull_order_to_array_bw(order, canvas, nails, strength):
    for pull_start, pull_end in zip(order, order[1:]):
        rr, cc, val = line_aa(nails[pull_start][0], nails[pull_start][1],
                              nails[pull_end][0], nails[pull_end][1])
        canvas[rr, cc] += val * strength
    return np.clip(canvas, a_min=0, a_max=1)

def process_single_variant(orig_pic, nails, shape, variant_name, image_dimens, 
                          wb=False, pull_amount=None, export_strength=0.1, random_nails=None,
                          radius1_multiplier=1, radius2_multiplier=1):
    logger.info(f"=== Processing {variant_name} variant ===")
    
    if wb:
        str_pic = init_canvas(shape, black=True)
        pull_order = create_art(nails, orig_pic, str_pic, 0.05, i_limit=pull_amount, random_nails=random_nails)
        blank = init_canvas(image_dimens, black=True)
    else:
        str_pic = init_canvas(shape, black=False)
        pull_order = create_art(nails, orig_pic, str_pic, -0.05, i_limit=pull_amount, random_nails=random_nails)
        blank = init_canvas(image_dimens, black=False)

    scaled_nails = scale_nails(
        image_dimens[1] / shape[1],
        image_dimens[0] / shape[0],
        nails
    )

    result = pull_order_to_array_bw(
        pull_order,
        blank,
        scaled_nails,
        export_strength if wb else -export_strength
    )
    
    return result, pull_order

def numpy_to_base64(image_array):
    """Convert numpy array to base64 encoded PNG string with memory management"""
    buffer = BytesIO()
    
    try:
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image_array, cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return image_base64
        
    finally:
        buffer.close()
        plt.close('all')
        gc.collect()

# ============== ASYNC JOB PROCESSING ==============

def process_string_art_job(job_id: str, image_data: bytes, params: Dict[str, Any]):
    """Background task to process string art generation"""
    try:
        logger.info(f"[{job_id}] Starting background processing")
        jobs_store[job_id]["status"] = JobStatus.PROCESSING
        jobs_store[job_id]["started_at"] = datetime.now().isoformat()
        
        # Process the image (same logic as before)
        pil_image = Image.open(BytesIO(image_data))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        img = np.array(pil_image)
        
        if np.any(img > 100):
            img = img / 255
        
        LONG_SIDE = 300
        
        if params.get('radius1_multiplier', 1) == 1 and params.get('radius2_multiplier', 1) == 1:
            img = largest_square(img)
            img = resize(img, (LONG_SIDE, LONG_SIDE))

        shape = (len(img), len(img[0]))

        # Create nails with fixed 200 count for circles
        if params.get('rect', False):
            nails = create_rectangle_nail_positions(shape, params.get('nail_step', 4))
            nail_labels = [str(i) for i in range(len(nails))]  # Keep numeric for rectangles
        else:
            nails = create_circle_nail_positions(shape, nail_count=200,
                                               r1_multip=params.get('radius1_multiplier', 1), 
                                               r2_multip=params.get('radius2_multiplier', 1))
            nail_labels = create_nail_labels(200)  # A1-A50, B1-B50, C1-C50, D1-D50

        logger.info(f"[{job_id}] Nails amount: {len(nails)}")
        logger.info(f"[{job_id}] Nail labels: {nail_labels[:10]}...{nail_labels[-10:]}")  # Show first and last 10
        
        # Convert to grayscale
        base_grayscale = rgb2gray(img) * 0.9
        
        # Define EXACTLY 7 variants with lower contrast levels (medium as maximum)
        variants = {
            'ultra_soft': {
                'process': lambda img: adjust_contrast(img, 0.25),
                'description': 'Very subtle details'
            },
            'very_low': {
                'process': lambda img: adjust_contrast(img, 0.35),
                'description': 'Gentle appearance'
            },
            'low_contrast': {
                'process': lambda img: adjust_contrast(img, 0.45),
                'description': 'Soft look'
            },
            'low_bright': {
                'process': lambda img: adjust_brightness(adjust_contrast(img, 0.5), -0.05),
                'description': 'Enhanced softness'
            },
            'medium_low': {
                'process': lambda img: adjust_contrast(img, 0.6),
                'description': 'Balanced softness'
            },
            'medium_soft': {
                'process': lambda img: adjust_contrast(img, 0.7),
                'description': 'Refined details'
            },
            'medium': {
                'process': lambda img: adjust_contrast(img, 0.75),
                'description': 'Maximum level - balanced and clear'
            }
        }
        
        side_len = params.get('side_len', 300)
        image_dimens = (int(side_len * params.get('radius1_multiplier', 1)), 
                       int(side_len * params.get('radius2_multiplier', 1)))
        
        results = {}
        
        # Process each variant
        for i, (variant_name, variant_config) in enumerate(variants.items(), 1):
            logger.info(f"[{job_id}] Processing variant {i}/7: {variant_name}")
            
            # Apply the variant's image processing
            processed_grayscale = variant_config['process'](base_grayscale)
            
            result, pull_order = process_single_variant(
                processed_grayscale, nails, shape, variant_name, image_dimens,
                wb=params.get('wb', False), 
                pull_amount=params.get('pull_amount'),
                export_strength=params.get('export_strength', 0.1),
                random_nails=params.get('random_nails'),
                radius1_multiplier=params.get('radius1_multiplier', 1),
                radius2_multiplier=params.get('radius2_multiplier', 1)
            )
            
            # Convert result to base64
            image_base64 = numpy_to_base64(result)
            
            # Convert pull order to labeled format
            pull_order_labeled = pull_order_to_labels(pull_order, nail_labels)
            pull_order_str = "-".join(pull_order_labeled) if pull_order_labeled else ""
            
            results[variant_name] = {
                "image_base64": image_base64,
                "pull_order": pull_order_str,
                "pull_order_numeric": "-".join([str(idx) for idx in pull_order]) if pull_order else "",  # Keep numeric version too
                "total_pulls": len(pull_order),
                "description": variant_config['description'],
                "variant_number": i
            }
            
            # Clean up memory
            del result, processed_grayscale
            gc.collect()
        
        # Update job with results
        jobs_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "completed_at": datetime.now().isoformat(),
            "results": results,
            "metadata": {
                "nails_count": len(nails),
                "image_dimensions": image_dimens,
                "original_shape": shape,
                "processing_params": params,
                "nail_labels": nail_labels,  # Include nail labels in metadata
                "nail_system": "sectioned" if not params.get('rect', False) else "numeric",
                "total_variants": 7,
                "variant_names": list(variants.keys())
            }
        })
        
        logger.info(f"[{job_id}] Job completed successfully with 7 variants")
        
    except Exception as e:
        logger.error(f"[{job_id}] Job failed: {str(e)}")
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
    return {"message": "String Art API is running", "status": "healthy", "variants": 7}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time(), "active_jobs": len(jobs_store), "variants_per_job": 7}

@app.post("/jobs")
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    side_len: int = 300,
    export_strength: float = 0.1,
    pull_amount: Optional[int] = None,
    random_nails: Optional[int] = None,
    radius1_multiplier: float = 1.0,
    radius2_multiplier: float = 1.0,
    nail_step: int = 4,
    wb: bool = False,
    rect: bool = False
):
    """
    Submit a string art generation job - Generates exactly 7 variants!
    Returns job_id immediately for polling
    
    7 Variants (lower contrast focus, medium as maximum):
    1. Ultra Soft (0.25 contrast) - Very subtle details
    2. Very Low (0.35 contrast) - Gentle appearance  
    3. Low Contrast (0.45 contrast) - Soft look
    4. Low Bright (0.5 contrast + brightness reduction) - Enhanced softness
    5. Medium Low (0.6 contrast) - Balanced softness
    6. Medium Soft (0.7 contrast) - Refined details
    7. Medium (0.75 contrast) - Maximum level - balanced and clear
    
    Circle patterns use sectioned nail labels: A1-A50, B1-B50, C1-C50, D1-D50
    Rectangle patterns use numeric labels: 0, 1, 2, ...
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Store job parameters
        params = {
            "side_len": side_len,
            "export_strength": export_strength,
            "pull_amount": pull_amount,
            "random_nails": random_nails,
            "radius1_multiplier": radius1_multiplier,
            "radius2_multiplier": radius2_multiplier,
            "nail_step": nail_step,
            "wb": wb,
            "rect": rect
        }
        
        # Initialize job in store
        jobs_store[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "filename": file.filename,
            "file_size": len(image_data),
            "params": params
        }
        
        # Start background processing
        background_tasks.add_task(process_string_art_job, job_id, image_data, params)
        
        logger.info(f"[{job_id}] Job submitted successfully")
        
        return {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "message": "Job submitted successfully. Will generate exactly 7 variants with optimized lower contrast processing.",
            "estimated_time": "3-7 minutes (7 variants)",
            "variants_count": 7,
            "nail_system": "Circle: A1-D50 sections, Rectangle: numeric"
        }
        
    except Exception as e:
        logger.error(f"Error submitting job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Check the status of a string art generation job
    Returns exactly 7 variants when completed
    """
    
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_store[job_id]
    
    # Return different responses based on status
    if job["status"] == JobStatus.COMPLETED:
        return {
            "job_id": job_id,
            "status": job["status"],
            "completed_at": job.get("completed_at"),
            "results": job["results"],
            "metadata": job["metadata"],
            "variants_count": len(job["results"])
        }
    elif job["status"] == JobStatus.FAILED:
        return {
            "job_id": job_id,
            "status": job["status"],
            "error": job.get("error"),
            "failed_at": job.get("failed_at")
        }
    else:
        # PENDING or PROCESSING
        return {
            "job_id": job_id,
            "status": job["status"],
            "created_at": job["created_at"],
            "started_at": job.get("started_at"),
            "message": "Job is processing 7 variants. Please check again in a few seconds.",
            "expected_variants": 7
        }

@app.get("/jobs")
async def list_jobs():
    """List all jobs (for debugging)"""
    return {
        "total_jobs": len(jobs_store),
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "created_at": job["created_at"],
                "filename": job.get("filename"),
                "variants": len(job.get("results", {})) if job.get("results") else 0
            }
            for job_id, job in jobs_store.items()
        ]
    }

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job from memory"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs_store[job_id]
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/variants")
async def get_variant_info():
    """Get information about the exactly 7 variants generated"""
    return {
        "total_variants": 7,
        "contrast_philosophy": "Focused on lower contrast levels for optimal string art results",
        "max_contrast_level": "medium (0.75)",
        "variants": {
            "ultra_soft": "Ultra Soft (0.25 contrast) - Very subtle details",
            "very_low": "Very Low (0.35 contrast) - Gentle appearance",
            "low_contrast": "Low Contrast (0.45 contrast) - Soft look",
            "low_bright": "Low Bright (0.5 contrast + brightness reduction) - Enhanced softness", 
            "medium_low": "Medium Low (0.6 contrast) - Balanced softness",
            "medium_soft": "Medium Soft (0.7 contrast) - Refined details",
            "medium": "Medium (0.75 contrast) - Maximum level - balanced and clear"
        },
        "processing_note": "All variants use contrast levels from 0.25 to 0.75 for optimal string art results"
    }

@app.get("/nail-labels")
async def get_nail_labels():
    """Get the nail labeling system for reference"""
    return {
        "circle_nails": {
            "total": 200,
            "sections": {
                "A": "A1 to A50 (nails 0-49)",
                "B": "B1 to B50 (nails 50-99)", 
                "C": "C1 to C50 (nails 100-149)",
                "D": "D1 to D50 (nails 150-199)"
            },
            "sample_labels": create_nail_labels(200)[:20] + ["..."] + create_nail_labels(200)[-20:]
        },
        "rectangle_nails": "Uses numeric labels: 0, 1, 2, ... (variable count based on nail_step)"
    }

# Legacy endpoint for backwards compatibility (but with timeout warning)
@app.post("/generate-string-art")
async def generate_string_art_sync(
    file: UploadFile = File(...),
    side_len: int = 300,
    export_strength: float = 0.1,
    pull_amount: Optional[int] = 1000,
    random_nails: Optional[int] = 50,
    radius1_multiplier: float = 1.0,
    radius2_multiplier: float = 1.0,
    nail_step: int = 4,
    wb: bool = False,
    rect: bool = False
):
    """
    DEPRECATED: Synchronous endpoint - may timeout on large images
    Use /jobs endpoint instead for reliable processing
    """
    
    return {
        "error": "This endpoint is deprecated due to timeout issues",
        "message": "Please use the async job endpoints instead:",
        "instructions": {
            "step_1": "POST /jobs - Submit your job and get a job_id",
            "step_2": "GET /jobs/{job_id} - Poll this endpoint until status is 'completed'",
            "step_3": "The completed job will contain exactly 7 variants in your results"
        },
        "recommended_flow": "Submit job → Poll status → Get 7 variants",
        "variants_count": 7,
        "nail_system": "Circle patterns now use A1-D50 sectioned labels"
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("String Art API server started successfully")
    logger.info("Now generating exactly 7 optimized variants per job!")
    logger.info("Focus: Lower contrast levels (0.25-0.75) for best string art results")
    logger.info("Available endpoints:")
    logger.info("  POST /jobs - Submit string art job (7 variants)")
    logger.info("  GET /jobs/{job_id} - Check job status")
    logger.info("  GET /jobs - List all jobs")
    logger.info("  GET /variants - View variant information")
    logger.info("  GET /nail-labels - View nail labeling system")
    logger.info("Nail system: Circle patterns use A1-D50 sections, rectangles use numeric")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("String Art API server shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")