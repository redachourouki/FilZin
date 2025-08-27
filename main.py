from fastapi import FastAPI, File, UploadFile, HTTPException
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
from typing import Dict, Any
import gc  # For garbage collection
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="String Art API", description="Generate string art from images")

# Enable CORS for Bubble integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== ORIGINAL ALGORITHM CODE (UNCHANGED) ==============

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def adjust_contrast(image, contrast_factor):
    """
    Adjust the contrast of a grayscale image.
    contrast_factor: 
    - < 1.0 for low contrast
    - = 1.0 for no change
    - > 1.0 for high contrast
    """
    # Apply contrast adjustment using the formula: new_pixel = (old_pixel - 0.5) * contrast + 0.5
    adjusted = (image - 0.5) * contrast_factor + 0.5
    # Clip values to valid range [0, 1]
    return np.clip(adjusted, 0, 1)

def largest_square(image: np.ndarray) -> np.ndarray:
    short_edge = np.argmin(image.shape[:2])  # 0 = vertical <= horizontal; 1 = otherwise
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

def create_circle_nail_positions(shape, nail_step=2, r1_multip=1, r2_multip=1):
    height = shape[0]
    width = shape[1]

    centre = (height // 2, width // 2)
    radius = min(height, width) // 2 - 1
    rr, cc = ellipse_perimeter(centre[0], centre[1], int(radius*r1_multip), int(radius*r2_multip))
    nails = list(set([(rr[i], cc[i]) for i in range(len(cc))]))
    nails.sort(key=lambda c: atan2(c[0] - centre[0], c[1] - centre[1]))
    nails = nails[::nail_step]

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

        cumulative_improvement =  np.sum(before_overlayed_line_diff - after_overlayed_line_diff)

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
    logger.info(f"DEBUG create_art: Returning pull_order with {len(pull_order)} pulls")
    return pull_order

def scale_nails(x_ratio, y_ratio, nails):
    return [(int(y_ratio*nail[0]), int(x_ratio*nail[1])) for nail in nails]

def pull_order_to_array_bw(order, canvas, nails, strength):
    # Draw a black and white pull order on the defined resolution
    for pull_start, pull_end in zip(order, order[1:]):  # pairwise iteration
        rr, cc, val = line_aa(nails[pull_start][0], nails[pull_start][1],
                              nails[pull_end][0], nails[pull_end][1])
        canvas[rr, cc] += val * strength

    return np.clip(canvas, a_min=0, a_max=1)

def process_single_contrast(orig_pic, nails, shape, contrast_label, image_dimens, 
                          wb=False, pull_amount=None, export_strength=0.1, random_nails=None,
                          radius1_multiplier=1, radius2_multiplier=1):
    """Process a single contrast level and return the result"""
    logger.info(f"=== Processing {contrast_label} contrast ===")
    
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

# ============== API WRAPPER CODE ==============

def numpy_to_base64(image_array):
    """Convert numpy array to base64 encoded PNG string with memory management"""
    buffer = BytesIO()
    
    try:
        # Create figure with explicit memory management
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image_array, cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)  # Explicitly close the figure
        
        # Get the PNG data and encode as base64
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return image_base64
        
    finally:
        buffer.close()
        plt.close('all')  # Close all matplotlib figures
        gc.collect()  # Force garbage collection

@app.get("/")
async def root():
    return {"message": "String Art API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time()}

@app.post("/generate-string-art")
async def generate_string_art(
    file: UploadFile = File(...),
    side_len: int = 300,
    export_strength: float = 0.1,
    pull_amount: int = None,
    random_nails: int = None,
    radius1_multiplier: float = 1.0,
    radius2_multiplier: float = 1.0,
    nail_step: int = 4,
    wb: bool = False,
    rect: bool = False
) -> Dict[str, Any]:
    """
    Generate string art from uploaded image with three contrast levels
    """
    
    request_id = str(uuid.uuid4())[:8]  # Short unique ID for this request
    logger.info(f"[{request_id}] Starting string art generation")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read uploaded image directly from memory
        contents = await file.read()
        logger.info(f"[{request_id}] Image file read, size: {len(contents)} bytes")
        
        try:
            # Use PIL to read image from memory, then convert to numpy array
            pil_image = Image.open(BytesIO(contents))
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert to numpy array (0-255 range)
            img = np.array(pil_image)
            logger.info(f"[{request_id}] Image processed, shape: {img.shape}")
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Normalize image if needed
        if np.any(img > 100):
            img = img / 255
        
        # Process image same as original code
        LONG_SIDE = 300
        
        if radius1_multiplier == 1 and radius2_multiplier == 1:
            img = largest_square(img)
            img = resize(img, (LONG_SIDE, LONG_SIDE))

        shape = (len(img), len(img[0]))

        # Create nails
        if rect:
            nails = create_rectangle_nail_positions(shape, nail_step)
        else:
            nails = create_circle_nail_positions(shape, nail_step, radius1_multiplier, radius2_multiplier)

        logger.info(f"[{request_id}] Nails amount: {len(nails)}")
        
        # Convert to grayscale
        base_grayscale = rgb2gray(img) * 0.9
        
        # Define contrast levels (same as original)
        contrast_levels = {
            'low': 0.5,    # Low contrast
            'mid': 0.75,   # Medium contrast
            'high': 0.87   # High contrast
        }
        
        image_dimens = int(side_len * radius1_multiplier), int(side_len * radius2_multiplier)
        
        results = {}
        
        # Process each contrast level
        for contrast_name, contrast_factor in contrast_levels.items():
            logger.info(f"[{request_id}] === Processing {contrast_name} contrast ===")
            
            # Apply contrast adjustment
            adjusted_grayscale = adjust_contrast(base_grayscale, contrast_factor)
            
            # Process with the existing algorithm (completely unchanged)
            result, pull_order = process_single_contrast(
                adjusted_grayscale, nails, shape, contrast_name, image_dimens,
                wb=wb, pull_amount=pull_amount, export_strength=export_strength,
                random_nails=random_nails, radius1_multiplier=radius1_multiplier,
                radius2_multiplier=radius2_multiplier
            )
            
            logger.info(f"[{request_id}] Pull order length: {len(pull_order) if pull_order else 0}")
            
            # Convert result to base64
            image_base64 = numpy_to_base64(result)
            
            # Ensure pull_order is properly formatted
            pull_order_str = "-".join([str(idx) for idx in pull_order]) if pull_order else ""
            
            results[contrast_name] = {
                "image_base64": image_base64,
                "pull_order": pull_order_str,
                "total_pulls": len(pull_order)
            }
            
            logger.info(f"[{request_id}] Completed {contrast_name} contrast processing")
            
            # Clean up memory after each contrast level
            del result, adjusted_grayscale
            gc.collect()
        
        response_data = {
            "success": True,
            "message": "String art generated successfully",
            "request_id": request_id,
            "results": results,
            "metadata": {
                "nails_count": len(nails),
                "image_dimensions": image_dimens,
                "original_shape": shape,
                "processing_params": {
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
            }
        }
        
        logger.info(f"[{request_id}] Returning successful response")
        return JSONResponse(content=response_data, status_code=200)
        
    except HTTPException:
        logger.error(f"[{request_id}] HTTP exception occurred")
        raise
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Clean up memory
        gc.collect()
        logger.info(f"[{request_id}] Request processing completed")

# Add a startup event to log server start
@app.on_event("startup")
async def startup_event():
    logger.info("String Art API server started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("String Art API server shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")