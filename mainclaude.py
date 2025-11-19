from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import matplotlib.image as mpimg
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


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="String Art API - Enhanced", description="Professional-grade string art generation")


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


# ============== NAIL LABELING SYSTEM ==============


def create_nail_labels(nail_count=200):
    """Create sectioned nail labels: A1-A50, B1-B50, C1-C50, D1-D50"""
    labels = []
    sections = ['A', 'B', 'C', 'D']
    nails_per_section = nail_count // 4
    
    for section_idx, section in enumerate(sections):
        for nail_num in range(1, nails_per_section + 1):
            labels.append(f"{section}{nail_num}")
    
    return labels


def index_to_label(index, nail_labels):
    """Convert numeric index to nail label"""
    if 0 <= index < len(nail_labels):
        return nail_labels[index]
    return str(index)


def pull_order_to_labels(pull_order, nail_labels):
    """Convert numeric pull order to labeled pull order"""
    return [index_to_label(idx, nail_labels) for idx in pull_order]


# ============== ENHANCED IMAGE PROCESSING ==============


def rgb2gray(rgb):
    """Convert RGB to grayscale using standard weights"""
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def gentle_contrast_enhancement(image, factor=1.15):
    """
    Gentle contrast enhancement - much more subtle than before
    Professional string art needs GENTLE preprocessing, not aggressive
    """
    mean_val = np.mean(image)
    adjusted = (image - mean_val) * factor + mean_val
    return np.clip(adjusted, 0, 1)


def adaptive_histogram_equalization(image, clip_limit=0.01):
    """
    CLAHE with very gentle clipping for subtle enhancement
    """
    # Convert to uint8 for CLAHE
    img_uint8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    enhanced = clahe.apply(img_uint8)
    return enhanced.astype(np.float64) / 255.0


def bilateral_filter_smooth(image, d=5, sigma_color=0.03, sigma_space=5):
    """
    Bilateral filtering to preserve edges while smoothing
    Much gentler than before - this is KEY for portraits
    """
    img_uint8 = (image * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(img_uint8, d, sigma_color * 255, sigma_space)
    return filtered.astype(np.float64) / 255.0


def downsample_for_perception(image, downsample_factor=2):
    """
    Downsample to simulate human visual blurring
    CRITICAL: This simulates how humans perceive string art from a distance
    """
    if downsample_factor <= 1:
        return image
    
    # Resize down
    new_shape = (image.shape[0] // downsample_factor, 
                 image.shape[1] // downsample_factor)
    downsampled = resize(image, new_shape, anti_aliasing=True)
    
    # Resize back up to original size for algorithm
    return resize(downsampled, image.shape, anti_aliasing=False)


def preprocess_portrait(image):
    """
    Professional portrait preprocessing pipeline
    Based on research: GENTLE is better than AGGRESSIVE
    """
    # Step 1: Bilateral filtering to smooth while preserving edges
    smoothed = bilateral_filter_smooth(image, d=5, sigma_color=0.03, sigma_space=5)
    
    # Step 2: Very gentle CLAHE
    enhanced = adaptive_histogram_equalization(smoothed, clip_limit=0.01)
    
    # Step 3: Subtle contrast boost (much less than before!)
    contrasted = gentle_contrast_enhancement(enhanced, factor=1.10)
    
    # Step 4: Downsample to simulate perception blur
    perception_blurred = downsample_for_perception(contrasted, downsample_factor=2)
    
    # Step 5: Final gentle gaussian blur
    final = gaussian_filter(perception_blurred, sigma=0.5)
    
    return np.clip(final, 0, 1)


def preprocess_general(image):
    """
    General image preprocessing - still gentle
    """
    # Gentle smoothing
    smoothed = gaussian_filter(image, sigma=0.7)
    
    # Subtle contrast
    contrasted = gentle_contrast_enhancement(smoothed, factor=1.12)
    
    # Perception blur
    perception_blurred = downsample_for_perception(contrasted, downsample_factor=2)
    
    return np.clip(perception_blurred, 0, 1)


def largest_square(image: np.ndarray) -> np.ndarray:
    """Extract largest square from image"""
    short_edge = np.argmin(image.shape[:2])
    short_edge_half = image.shape[short_edge] // 2
    long_edge_center = image.shape[1 - short_edge] // 2
    if short_edge == 0:
        return image[:, long_edge_center - short_edge_half:
                        long_edge_center + short_edge_half]
    if short_edge == 1:
        return image[long_edge_center - short_edge_half:
                     long_edge_center + short_edge_half, :]


# ============== NAIL POSITIONING ==============


def create_circle_nail_positions(shape, nail_count=200, r1_multip=1, r2_multip=1):
    """Create circular nail positions"""
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
    """Initialize canvas"""
    if black:
        return np.zeros(shape)
    else:
        return np.ones(shape)


def get_aa_line(from_pos, to_pos, str_strength, picture):
    """Get anti-aliased line with given strength"""
    rr, cc, val = line_aa(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
    line = picture[rr, cc] + str_strength * val
    line = np.clip(line, a_min=0, a_max=1)
    return line, rr, cc


# ============== ENHANCED CORE ALGORITHM ==============


def find_best_nail_position_fearless(current_position, nails, str_pic, orig_pic, 
                                      str_strength, random_nails=None):
    """
    FEARLESS error calculation - KEY IMPROVEMENT from research!
    
    Only penalize if the line makes things WORSE
    Don't penalize if it makes things BETTER (even if it temporarily worsens other areas)
    
    This allows the algorithm to be "fearless" - it can temporarily mess up areas
    knowing that future strings will fix them
    """
    best_cumulative_improvement = -99999
    best_nail_position = None
    best_nail_idx = None
    
    if random_nails is not None:
        nail_ids = np.random.choice(range(len(nails)), size=random_nails, replace=False)
        nails_and_ids = list(zip(nail_ids, nails[nail_ids]))
    else:
        nails_and_ids = enumerate(nails)

    for nail_idx, nail_position in nails_and_ids:
        # Get the proposed line
        overlayed_line, rr, cc = get_aa_line(current_position, nail_position, str_strength, str_pic)
        
        # Calculate pixel-wise improvement
        before_diff = np.abs(str_pic[rr, cc] - orig_pic[rr, cc])
        after_diff = np.abs(overlayed_line - orig_pic[rr, cc])
        
        # FEARLESS: Only count pixels that IMPROVE
        # This is the key insight from Michael Crum's research
        pixel_improvements = before_diff - after_diff
        
        # Only sum positive improvements (pixels that got better)
        # Ignore pixels that got worse (they'll be fixed later)
        positive_improvements = np.maximum(pixel_improvements, 0)
        cumulative_improvement = np.sum(positive_improvements)

        if cumulative_improvement >= best_cumulative_improvement:
            best_cumulative_improvement = cumulative_improvement
            best_nail_position = nail_position
            best_nail_idx = nail_idx

    return best_nail_idx, best_nail_position, best_cumulative_improvement


def create_art_enhanced(nails, orig_pic, str_pic, str_strength, 
                       min_iterations=6000, max_iterations=15000, 
                       max_fails=10, random_nails=None):
    """
    ENHANCED string art algorithm with:
    1. Much lighter string strength (0.01-0.02 instead of 0.05)
    2. Fearless error calculation
    3. Minimum iteration requirement (6000+)
    4. Extended failure tolerance (10 consecutive fails)
    5. Better stopping conditions
    
    Based on research from:
    - Michael Crum's string art generator
    - Birsak et al. "String Art: Towards Computational Fabrication"
    - Various academic papers on greedy optimization
    """
    start = time()
    iter_times = []
    current_position = nails[0]
    pull_order = [0]
    i = 0
    consecutive_fails = 0
    last_improvement = 0
    
    logger.info(f"Starting enhanced algorithm with min_iterations={min_iterations}, max_iterations={max_iterations}")
    logger.info(f"String strength: {str_strength}")
    
    while True:
        start_iter = time()
        i += 1
        
        if i % 500 == 0:
            logger.info(f"Iteration {i} - Total pulls: {len(pull_order)} - Consecutive fails: {consecutive_fails}")
        
        # Stopping conditions
        if i >= max_iterations:
            logger.info(f"Reached maximum iterations: {max_iterations}")
            break
            
        if i >= min_iterations and consecutive_fails >= max_fails:
            logger.info(f"Reached minimum iterations ({min_iterations}) and {consecutive_fails} consecutive fails")
            break

        # Find best nail with FEARLESS error calculation
        idx, best_nail_position, best_cumulative_improvement = find_best_nail_position_fearless(
            current_position, nails, str_pic, orig_pic, str_strength, random_nails
        )

        # Check if we found an improvement
        if best_cumulative_improvement <= 0:
            consecutive_fails += 1
            continue
        
        # Reset consecutive fails counter
        consecutive_fails = 0
        
        # Apply the string
        pull_order.append(idx)
        best_overlayed_line, rr, cc = get_aa_line(current_position, best_nail_position, str_strength, str_pic)
        str_pic[rr, cc] = best_overlayed_line
        current_position = best_nail_position
        last_improvement = best_cumulative_improvement
        
        iter_times.append(time() - start_iter)

    logger.info(f"Algorithm completed!")
    logger.info(f"Total time: {time() - start:.2f}s")
    logger.info(f"Total pulls: {len(pull_order)}")
    logger.info(f"Average iteration time: {np.mean(iter_times) if iter_times else 0:.4f}s")
    
    return pull_order


def scale_nails(x_ratio, y_ratio, nails):
    """Scale nail positions"""
    return [(int(y_ratio*nail[0]), int(x_ratio*nail[1])) for nail in nails]


def pull_order_to_array_bw(order, canvas, nails, strength):
    """Render pull order to array"""
    for pull_start, pull_end in zip(order, order[1:]):
        rr, cc, val = line_aa(nails[pull_start][0], nails[pull_start][1],
                              nails[pull_end][0], nails[pull_end][1])
        canvas[rr, cc] += val * strength
    return np.clip(canvas, a_min=0, a_max=1)


def process_single_variant(orig_pic, nails, shape, variant_name, image_dimens, 
                          wb=False, export_strength=0.08, random_nails=None,
                          radius1_multiplier=1, radius2_multiplier=1):
    """Process a single variant with enhanced algorithm"""
    logger.info(f"=== Processing {variant_name} variant (ENHANCED) ===")
    
    if wb:
        # White string on black background
        str_pic = init_canvas(shape, black=True)
        # CRITICAL: Much lighter string strength!
        pull_order = create_art_enhanced(
            nails, orig_pic, str_pic, 
            str_strength=0.015,  # Was 0.05, now 0.015
            min_iterations=6000,
            max_iterations=15000,
            max_fails=10,
            random_nails=random_nails
        )
        blank = init_canvas(image_dimens, black=True)
    else:
        # Black string on white background  
        str_pic = init_canvas(shape, black=False)
        # CRITICAL: Much lighter string strength!
        pull_order = create_art_enhanced(
            nails, orig_pic, str_pic,
            str_strength=-0.015,  # Was -0.05, now -0.015
            min_iterations=6000,
            max_iterations=15000,
            max_fails=10,
            random_nails=random_nails
        )
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
    """Convert numpy array to base64 encoded PNG"""
    buffer = BytesIO()
    
    try:
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image_array, cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
        
        buffer.seek(0)
        import base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return image_base64
        
    finally:
        buffer.close()
        plt.close('all')
        gc.collect()


# ============== ASYNC JOB PROCESSING ==============


def process_string_art_job(job_id: str, image_data: bytes, params: Dict[str, Any]):
    """Background task to process string art generation with ENHANCED algorithm"""
    try:
        logger.info(f"[{job_id}] Starting ENHANCED processing")
        jobs_store[job_id]["status"] = JobStatus.PROCESSING
        jobs_store[job_id]["started_at"] = datetime.now().isoformat()
        
        # Load image
        pil_image = Image.open(BytesIO(image_data))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        img = np.array(pil_image)
        
        if np.any(img > 100):
            img = img / 255
        
        LONG_SIDE = 300
        
        img = largest_square(img)
        img = resize(img, (LONG_SIDE, LONG_SIDE))
        shape = (len(img), len(img[0]))

        # Create nails (200 for circles)
        nails = create_circle_nail_positions(shape, nail_count=200,
                                           r1_multip=params.get('radius1_multiplier', 1), 
                                           r2_multip=params.get('radius2_multiplier', 1))
        nail_labels = create_nail_labels(200)

        logger.info(f"[{job_id}] Nails: {len(nails)}")
        
        # Convert to grayscale
        base_grayscale = rgb2gray(img)
        
        # Define 3 professional preprocessing variants
        # CRITICAL: All variants use GENTLE preprocessing now!
        variants = {
            'portrait_optimized': {
                'process': lambda img: preprocess_portrait(img * 0.92),
                'description': 'Professional portrait preset - bilateral filtering + gentle CLAHE',
                'export_strength': 0.08
            },
            'general_balanced': {
                'process': lambda img: preprocess_general(img * 0.90),
                'description': 'General images - gentle smoothing + subtle contrast',
                'export_strength': 0.08
            },
            'minimal_processing': {
                'process': lambda img: gaussian_filter(img * 0.88, sigma=0.8),
                'description': 'Minimal preprocessing - just gentle blur',
                'export_strength': 0.08
            }
        }
        
        side_len = params.get('side_len', 300)
        image_dimens = (int(side_len * params.get('radius1_multiplier', 1)), 
                       int(side_len * params.get('radius2_multiplier', 1)))
        
        results = {}
        
        # Process each variant
        for i, (variant_name, variant_config) in enumerate(variants.items(), 1):
            logger.info(f"[{job_id}] Processing variant {i}/3: {variant_name}")
            
            # Apply preprocessing
            processed_grayscale = variant_config['process'](base_grayscale)
            
            result, pull_order = process_single_variant(
                processed_grayscale, nails, shape, variant_name, image_dimens,
                wb=params.get('wb', False), 
                export_strength=variant_config['export_strength'],
                random_nails=params.get('random_nails'),
                radius1_multiplier=params.get('radius1_multiplier', 1),
                radius2_multiplier=params.get('radius2_multiplier', 1)
            )
            
            # Convert to base64
            image_base64 = numpy_to_base64(result)
            
            # Convert pull order to labels
            pull_order_labeled = pull_order_to_labels(pull_order, nail_labels)
            pull_order_str = "-".join(pull_order_labeled) if pull_order_labeled else ""
            
            results[variant_name] = {
                "image_base64": image_base64,
                "pull_order": pull_order_str,
                "pull_order_numeric": "-".join([str(idx) for idx in pull_order]) if pull_order else "",
                "total_pulls": len(pull_order),
                "description": variant_config['description'],
                "variant_number": i
            }
            
            # Clean up
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
                "nail_labels": nail_labels,
                "nail_system": "sectioned (A1-D50)",
                "total_variants": 3,
                "variant_names": list(variants.keys()),
                "algorithm_version": "enhanced_v2_professional",
                "key_improvements": [
                    "Fearless error calculation",
                    "Much lighter string strength (0.015 vs 0.05)",
                    "Minimum 6000 iterations",
                    "Extended failure tolerance (10 fails)",
                    "Gentle preprocessing (bilateral + CLAHE)",
                    "Perception downsampling"
                ]
            }
        })
        
        logger.info(f"[{job_id}] ENHANCED Job completed with 3 professional variants")
        
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
    return {
        "message": "ENHANCED String Art API - Professional Grade",
        "status": "healthy",
        "version": "2.0",
        "variants": 3,
        "improvements": [
            "Fearless error calculation",
            "6000-15000 iterations",
            "Lighter string strength (0.015)",
            "Gentle preprocessing",
            "Professional quality output"
        ]
    }


@app.post("/jobs")
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    side_len: int = 300,
    random_nails: Optional[int] = None,
    wb: bool = False
):
    """
    Submit ENHANCED string art generation job
    
    Returns 3 professional variants:
    1. Portrait Optimized - Bilateral filtering + gentle CLAHE
    2. General Balanced - Gentle smoothing + subtle contrast  
    3. Minimal Processing - Just gentle blur
    
    All use ENHANCED algorithm with:
    - Fearless error calculation
    - Light string strength (0.015)
    - 6000-15000 iterations
    - 10 consecutive fail tolerance
    """
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    job_id = str(uuid.uuid4())
    
    try:
        image_data = await file.read()
        
        params = {
            "side_len": side_len,
            "random_nails": random_nails,
            "wb": wb,
            "radius1_multiplier": 1.0,
            "radius2_multiplier": 1.0
        }
        
        jobs_store[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "filename": file.filename,
            "file_size": len(image_data),
            "params": params
        }
        
        background_tasks.add_task(process_string_art_job, job_id, image_data, params)
        
        logger.info(f"[{job_id}] ENHANCED job submitted")
        
        return {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "message": "ENHANCED job submitted - will generate 3 professional variants",
            "estimated_time": "10-20 minutes (6000-15000 iterations per variant)",
            "variants_count": 3,
            "algorithm": "enhanced_v2_professional"
        }
        
    except Exception as e:
        logger.error(f"Error submitting job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Check job status"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_store[job_id]
    
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
        return {
            "job_id": job_id,
            "status": job["status"],
            "created_at": job["created_at"],
            "started_at": job.get("started_at"),
            "message": "Processing with ENHANCED algorithm (6000-15000 iterations)",
            "expected_variants": 3
        }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0_enhanced",
        "algorithm": "fearless_greedy_with_perception_blur",
        "active_jobs": len(jobs_store)
    }


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 70)
    logger.info("ENHANCED String Art API v2.0 - Professional Grade")
    logger.info("=" * 70)
    logger.info("Key Improvements:")
    logger.info("  ✓ Fearless error calculation (only penalize negative improvements)")
    logger.info("  ✓ Much lighter string strength (0.015 vs 0.05)")
    logger.info("  ✓ Minimum 6000 iterations, max 15000")
    logger.info("  ✓ Extended failure tolerance (10 consecutive fails)")
    logger.info("  ✓ Gentle preprocessing (bilateral + CLAHE)")
    logger.info("  ✓ Perception downsampling for human-like blurring")
    logger.info("=" * 70)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")