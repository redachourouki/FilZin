from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa
from skimage.transform import resize
from time import time
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

app = FastAPI(title="String Art API (TEST MODE)", description="Generate string art from images without preprocessing")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== JOB SYSTEM =====

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

jobs_store: Dict[str, Dict] = {}

# ===== NAIL SYSTEM =====

def create_nail_labels(nail_count=200):
    labels = []
    sections = ['A', 'B', 'C', 'D']
    nails_per_section = nail_count // 4
    for section_idx, section in enumerate(sections):
        for nail_num in range(1, nails_per_section + 1):
            labels.append(f"{section}{nail_num}")
    return labels

def index_to_label(index, nail_labels):
    if 0 <= index < len(nail_labels):
        return nail_labels[index]
    return str(index)

def pull_order_to_labels(pull_order, nail_labels):
    return [index_to_label(idx, nail_labels) for idx in pull_order]

# ===== STRING ART FUNCTIONS =====

def largest_square(image: np.ndarray) -> np.ndarray:
    short_edge = np.argmin(image.shape[:2])
    short_edge_half = image.shape[short_edge] // 2
    long_edge_center = image.shape[1 - short_edge] // 2
    if short_edge == 0:
        return image[:, long_edge_center - short_edge_half: long_edge_center + short_edge_half]
    if short_edge == 1:
        return image[long_edge_center - short_edge_half: long_edge_center + short_edge_half, :]

def create_rectangle_nail_positions(shape, nail_step=2):
    height, width = shape
    nails_top = [(0, i) for i in range(0, width, nail_step)]
    nails_bot = [(height - 1, i) for i in range(0, width, nail_step)]
    nails_right = [(i, width - 1) for i in range(1, height - 1, nail_step)]
    nails_left = [(i, 0) for i in range(1, height - 1, nail_step)]
    return np.array(nails_top + nails_right + nails_bot + nails_left)

def create_circle_nail_positions(shape, nail_count=200, r1_multip=1, r2_multip=1):
    height, width = shape
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
    return np.zeros(shape) if black else np.ones(shape)

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
        before_overlayed_line_diff = np.abs(str_pic[rr, cc] - orig_pic[rr, cc]) ** 2
        after_overlayed_line_diff = np.abs(overlayed_line - orig_pic[rr, cc]) ** 2
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
        if i % 500 == 0:
            logger.info(f"Iteration {i}")
        if i_limit is None:
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
    return [(int(y_ratio * nail[0]), int(x_ratio * nail[1])) for nail in nails]

def pull_order_to_array_bw(order, canvas, nails, strength):
    for pull_start, pull_end in zip(order, order[1:]):
        rr, cc, val = line_aa(nails[pull_start][0], nails[pull_start][1],
                              nails[pull_end][0], nails[pull_end][1])
        canvas[rr, cc] += val * strength
    return np.clip(canvas, a_min=0, a_max=1)

def process_single_variant(orig_pic, nails, shape, variant_name, image_dimens, wb=False, pull_amount=None, export_strength=0.1, random_nails=None, radius1_multiplier=1, radius2_multiplier=1):
    logger.info(f"=== Processing {variant_name} variant ===")

    if wb:
        str_pic = init_canvas(shape, black=True)
        pull_order = create_art(nails, orig_pic, str_pic, 0.05, i_limit=pull_amount, random_nails=random_nails)
        blank = init_canvas(image_dimens, black=True)
    else:
        str_pic = init_canvas(shape, black=False)
        pull_order = create_art(nails, orig_pic, str_pic, -0.05, i_limit=pull_amount, random_nails=random_nails)
        blank = init_canvas(image_dimens, black=False)

    scaled_nails = scale_nails(image_dimens[1] / shape[1], image_dimens[0] / shape[0], nails)
    result = pull_order_to_array_bw(pull_order, blank, scaled_nails, export_strength if wb else -export_strength)
    return result, pull_order

def numpy_to_base64(image_array):
    buffer = BytesIO()
    try:
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image_array, cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    finally:
        buffer.close()
        plt.close('all')
        gc.collect()

# ===== TEST MODE JOB FUNCTION =====

def process_string_art_job(job_id: str, image_data: bytes, params: Dict[str, Any]):
    """Background task to process string art generation (TEST MODE: no preprocessing or variants)."""
    try:
        print("TEST MODE ACTIVE: Skipping preprocessing & multiple variants")
        logger.info(f"[{job_id}] TEST MODE ACTIVE: Skipping preprocessing & multiple variants")

        jobs_store[job_id]["status"] = JobStatus.PROCESSING
        jobs_store[job_id]["started_at"] = datetime.now().isoformat()

        # --- Load uploaded image directly (NO preprocessing) ---
        pil_image = Image.open(BytesIO(image_data))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        img = np.array(pil_image)

        if np.any(img > 100):
            img = img / 255

        LONG_SIDE = 300
        img = resize(largest_square(img), (LONG_SIDE, LONG_SIDE))
        shape = (len(img), len(img[0]))

        # --- Create nails (same logic) ---
        if params.get('rect', False):
            nails = create_rectangle_nail_positions(shape, params.get('nail_step', 4))
            nail_labels = [str(i) for i in range(len(nails))]
        else:
            nails = create_circle_nail_positions(shape, nail_count=200,
                                                r1_multip=params.get('radius1_multiplier', 1),
                                                r2_multip=params.get('radius2_multiplier', 1))
            nail_labels = create_nail_labels(200)

        logger.info(f"[{job_id}] Nails amount: {len(nails)}")

        # --- Use uploaded image directly ---
        orig_pic = img.mean(axis=2) if img.ndim == 3 else img
        image_dimens = (int(params.get('side_len', 300) * params.get('radius1_multiplier', 1)),
                        int(params.get('side_len', 300) * params.get('radius2_multiplier', 1)))

        result, pull_order = process_single_variant(
            orig_pic, nails, shape, "test_variant", image_dimens,
            wb=params.get('wb', False),
            pull_amount=params.get('pull_amount'),
            export_strength=params.get('export_strength', 0.1),
            random_nails=params.get('random_nails'),
            radius1_multiplier=params.get('radius1_multiplier', 1),
            radius2_multiplier=params.get('radius2_multiplier', 1)
        )

        image_base64 = numpy_to_base64(result)
        pull_order_labeled = pull_order_to_labels(pull_order, nail_labels)
        pull_order_str = "-".join(pull_order_labeled)

        results = {
            "test_variant": {
                "image_base64": image_base64,
                "pull_order": pull_order_str,
                "pull_order_numeric": "-".join([str(idx) for idx in pull_order]),
                "total_pulls": len(pull_order),
                "description": "Single direct-processing test variant",
                "variant_number": 1
            }
        }

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
                "nail_system": "sectioned" if not params.get('rect', False) else "numeric",
                "total_variants": 1,
                "variant_names": ["test_variant"],
                "mode": "TEST MODE - NO PREPROCESSING"
            }
        })

        logger.info(f"[{job_id}] Job completed successfully in TEST MODE")

    except Exception as e:
        logger.error(f"[{job_id}] Job failed: {str(e)}")
        jobs_store[job_id].update({
            "status": JobStatus.FAILED,
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
    finally:
        gc.collect()

# ===== API ENDPOINTS =====

@app.get("/")
async def root():
    return {"message": "String Art API running in TEST MODE", "variants": 1}

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
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    job_id = str(uuid.uuid4())
    image_data = await file.read()

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

    jobs_store[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        "filename": file.filename,
        "file_size": len(image_data),
        "params": params
    }

    background_tasks.add_task(process_string_art_job, job_id, image_data, params)
    logger.info(f"[{job_id}] Job submitted successfully")

    return {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "message": "Job submitted in TEST MODE. Will generate one direct-processing variant.",
        "variants_count": 1
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_store[job_id]

@app.on_event("startup")
async def startup_event():
    logger.info("String Art API started in TEST MODE (no preprocessing, single variant)")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
