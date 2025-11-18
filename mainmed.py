# stringart_api.py
# Option B: Portrait-optimized string art API (FastAPI)
# Dependencies:
# pip install fastapi uvicorn numpy pillow scikit-image opencv-python scipy matplotlib

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from enum import Enum
from io import BytesIO
from datetime import datetime
import numpy as np
from PIL import Image, ImageOps
import cv2
from skimage.draw import line_aa
from skimage.transform import resize
import base64
import uuid
import logging
import math
import gc
import matplotlib.pyplot as plt

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stringart")

# ---------- app ----------
app = FastAPI(title="StringArt API - Option B (Portrait-Optimized)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- job store ----------
jobs_store: Dict[str, Dict] = {}

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# ---------- utilities ----------
def pil_to_np(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    return arr

def np_to_base64_png(arr: np.ndarray, cmap='gray'):
    buf = BytesIO()
    # arr assumed in [0..1]
    fig = plt.figure(figsize=(6,6), dpi=150)
    plt.imshow(arr, cmap=cmap, vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return b64

# ---------- nail helpers ----------
def create_nail_labels(nail_count=200):
    sections = ['A','B','C','D']
    per = nail_count // 4
    labels = []
    for i,sec in enumerate(sections):
        for n in range(1, per+1):
            labels.append(f"{sec}{n}")
    # if remainder
    extra = nail_count - len(labels)
    idx = 1
    while extra>0:
        labels.append(f"X{idx}")
        idx+=1
        extra-=1
    return labels

def create_circle_nail_positions(shape, nail_count=200, radius_shrink=0.98):
    h,w = shape
    cy, cx = h//2, w//2
    # radius in pixels
    radius = int(min(h,w) * 0.5 * radius_shrink)
    nails = []
    for i in range(nail_count):
        ang = 2*np.pi * i / nail_count
        y = int(round(cy + radius * math.sin(ang)))
        x = int(round(cx + radius * math.cos(ang)))
        nails.append((y,x))
    return np.asarray(nails, dtype=int)

# ---------- preprocessing pipeline (Option B) ----------
def preprocess_portrait(img_rgb: np.ndarray, side_len=400, do_face_boost=True):
    """
    Input: img_rgb float32 in [0..1] or uint8 0..255
    Returns: target grayscale float32 in [0..1] sized (side_len, side_len)
    Steps:
      - crop largest square, resize to side_len
      - bilateral smoothing to remove texture but preserve edges
      - CLAHE
      - face detection -> localized boosts (eyes / mouth)
      - multi-scale edge map
      - perceptual blend: combine smoothed image + edges (weights)
      - Floyd–Steinberg dithering to produce binary target suitable for solver
      - also return continuous importance map (for weighted scoring if needed)
    """
    if img_rgb.dtype != np.float32:
        img_rgb = img_rgb.astype(np.float32) / 255.0

    # crop largest centered square
    h,w,_ = img_rgb.shape
    short = min(h,w)
    y0 = (h - short)//2
    x0 = (w - short)//2
    img_sq = img_rgb[y0:y0+short, x0:x0+short, :]

    # resize
    img_res = cv2.resize((img_sq*255).astype(np.uint8), (side_len, side_len), interpolation=cv2.INTER_AREA)
    img_res_f = img_res.astype(np.float32) / 255.0

    # convert to gray
    gray = cv2.cvtColor((img_res_f*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # bilateral filter (remove texture, keep edges)
    bf = cv2.bilateralFilter((gray*255).astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)
    bf = bf.astype(np.float32) / 255.0

    # CLAHE (adaptive contrast)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    clahe_res = clahe.apply((bf*255).astype(np.uint8)).astype(np.float32) / 255.0

    enhanced = clahe_res.copy()

    # Face detection and local boosts (Haar cascade)
    if do_face_boost:
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale((img_res*255).astype(np.uint8), scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
            for (x,y,wf,hf) in faces:
                # convert box coords (x,y,w,h) to masks
                # eye region
                ey1 = int(y + 0.15*hf)
                ey2 = int(y + 0.45*hf)
                ex1 = int(x + 0.18*wf)
                ex2 = int(x + 0.82*wf)
                ey1 = max(0,ey1); ey2=min(side_len,ey2); ex1=max(0,ex1); ex2=min(side_len,ex2)
                enhanced[ey1:ey2, ex1:ex2] = np.minimum(1.0, enhanced[ey1:ey2, ex1:ex2]*1.35 + 0.12)
                # mouth region
                my1 = int(y + 0.6*hf)
                my2 = int(y + 0.85*hf)
                mx1 = int(x + 0.25*wf)
                mx2 = int(x + 0.75*wf)
                my1 = max(0,my1); my2=min(side_len,my2); mx1=max(0,mx1); mx2=min(side_len,mx2)
                enhanced[my1:my2, mx1:mx2] = np.minimum(1.0, enhanced[my1:my2, mx1:mx2]*1.25 + 0.08)
        except Exception as e:
            logger.info("face boost failed: " + str(e))

    # Multi-scale edges
    def multi_scale_edges(img_gray):
        edges_total = np.zeros_like(img_gray)
        sigmas = [0.7, 1.5, 3.0]
        weights = [0.45, 0.35, 0.20]
        for s,w in zip(sigmas, weights):
            k = int(max(3, round(s*3))*2+1)
            blurred = cv2.GaussianBlur(img_gray, (k,k), s)
            sobx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            soby = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
            edges = np.hypot(sobx, soby)
            edges = edges / (edges.max() + 1e-9)
            edges_total += edges * w
        edges_total = np.clip(edges_total, 0.0, 1.0)
        return edges_total

    edges = multi_scale_edges(enhanced)

    # Blend: keep important structural lines but maintain tonal info
    blended = 0.75*enhanced + 0.35*edges
    blended = np.clip(blended, 0.0, 1.0)

    # Final gamma to brighten slightly (good for prints)
    gamma = 0.85
    blended = np.power(blended, gamma)

    # Normalize
    blended = (blended - blended.min()) / (blended.max() - blended.min() + 1e-9)

    # Produce an "importance" map (higher means darker target -> more thread)
    # We want black thread on white board, so invert: darker = 1.0
    importance = 1.0 - blended

    # Floyd–Steinberg dithering to get binary map (keeps tonal gradation)
    def floyd_steinberg_dither(img):
        arr = (img*255).astype(np.int32)
        H,W = arr.shape
        out = np.zeros_like(arr, dtype=np.uint8)
        A = arr.astype(np.float32)
        for y in range(H):
            for x in range(W):
                old = A[y,x]
                new = 0 if old < 128 else 255
                out[y,x] = new
                quant_error = old - new
                if x+1 < W: A[y, x+1] += quant_error * 7/16
                if y+1 < H:
                    if x>0: A[y+1, x-1] += quant_error * 3/16
                    A[y+1, x] += quant_error * 5/16
                    if x+1 < W: A[y+1, x+1] += quant_error * 1/16
        return out.astype(np.uint8) / 255.0

    binary = floyd_steinberg_dither(importance)

    # final returns: binary target (for solver) and continuous importance map (for diagnostics)
    return binary.astype(np.float32), importance.astype(np.float32)

# ---------- mask precomputation (packed bits) ----------
def precompute_line_masks(nails, H, W):
    """
    Build boolean masks for each line (nail_i -> nail_j) packed as uint8 bytes.
    Returns:
      masks: numpy uint8 array of shape (num_lines, bytes_per_image)
      lines_map: list of (i,j)
      bytes_per_image = ceil(H*W / 8)
    """
    coords_count = H * W
    bytes_per_image = (coords_count + 7) // 8
    lines = []
    masks = []

    # precompute line indices only for directed moves? we will use undirected mapping from i->j
    n = len(nails)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # draw line antialiased -> produce fractional coverage; we'll threshold >0
            rr, cc, val = line_aa(nails[i][0], nails[i][1], nails[j][0], nails[j][1])
            indices = rr * W + cc
            # create empty byte array
            bytearr = np.zeros(bytes_per_image, dtype=np.uint8)
            for idx in np.unique(indices):
                byte_index = idx // 8
                bit_index = idx % 8
                bytearr[byte_index] |= (1 << bit_index)
            masks.append(bytearr)
            lines.append((i,j))
    masks = np.stack(masks, axis=0)  # shape (num_lines, bytes_per_image)
    return masks, lines

# popcount lookup (0..255)
_popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

def score_masks_with_residual(masks_uint8, residual_uint8):
    """
    masks_uint8: (L, B) uint8
    residual_uint8: (B,) uint8
    returns scores: (L,) int
    """
    # bitwise and
    # broadcasting residual across lines:
    anded = np.bitwise_and(masks_uint8, residual_uint8[None, :])
    # popcount per byte and sum across bytes
    # anded is small enough for memory; compute via lookup
    scores = _popcount[anded].sum(axis=1)
    return scores

# ---------- core solver (Option B) ----------
def solver_precomputed_mask(orig_binary, importance_map, nails, pull_amount=None, random_nails=None, min_distance_skip=2, debug=False):
    """
    orig_binary: desired black pixels (1.0 = black) shape H,W (float32 in {0,1})
    importance_map: continuous importance used as residual initial
    nails: array of (y,x)
    returns: pull_order (list of nail indices)
    Approach:
      - pack residual into uint8 bit array
      - precompute masks for each candidate line (i->j)
      - greedy iterative: at each step sample candidate subset, compute score via popcount, pick best positive improvement
      - update residual by zeroing bits covered by the chosen line (simulate darkening)
    """
    H,W = orig_binary.shape
    # flatten residual as binary uint8
    residual = (orig_binary > 0.5).astype(np.uint8).flatten()  # 0/1
    coords = H*W
    bytes_per_image = (coords + 7)//8
    residual_bytes = np.packbits(residual, bitorder='little')  # default packs along last axis; ensures correct bit positions
    # packbits returns array shape (bytes_per_image,) when input 1D
    # Precompute masks:
    logger.info("Precomputing line masks (this may take a few seconds)...")
    masks_uint8, lines_map = precompute_line_masks(nails, H, W)  # (L,B)
    logger.info(f"Precomputed {masks_uint8.shape[0]} line masks, bytes_per_image={masks_uint8.shape[1]}")
    # available lines mask: initially all valid (we won't exclude)
    num_lines = masks_uint8.shape[0]

    # iterative selection
    pull_order = []
    current_idx = 0  # start at nail 0
    pull_order.append(current_idx)
    iter_count = 0
    fails = 0
    max_iters = pull_amount if pull_amount else 4000
    # compute initial residual bytes
    res_bytes = residual_bytes.copy()

    # helper: compute score for candidate subset indices
    def score_subset(indices):
        if len(indices) == 0:
            return np.array([], dtype=int)
        subset_masks = masks_uint8[indices]
        scores = score_masks_with_residual(subset_masks, res_bytes)
        return scores

    # precompute distance matrix maybe used for skip neighbor constraint
    nail_positions = nails
    # compute pairwise angular/linear distances
    # use Euclidean distances
    dists = np.linalg.norm(nail_positions[:,None,:] - nail_positions[None,:,:], axis=2)

    while iter_count < max_iters:
        iter_count += 1
        # candidate selection:
        # by default sample a number of random lines (random_nails param) as candidates from all lines that start from current_idx
        # lines_map is mapping from line_id -> (i,j), we need lines that start from current_idx
        # build line ids starting at current_idx
        start_line_ids = []
        # compute start ids quickly: they occur at positions where i==current_idx
        # because precompute loops i in range(n) then j in range(n) we can compute direct slice
        n = len(nails)
        # compute offset where lines for current_idx start: current_idx * (n) but since self-lines removed, indexing is tricky.
        # simpler: we will gather indices via comprehension (cost minor vs mask precompute)
        for lid, (i,j) in enumerate(lines_map):
            if i == current_idx:
                start_line_ids.append(lid)
        if not start_line_ids:
            # fallback: allow any line
            candidate_ids = np.arange(num_lines)
        else:
            # apply random sampling among start_line_ids
            if random_nails and random_nails < len(start_line_ids):
                candidate_ids = np.random.choice(start_line_ids, size=random_nails, replace=False)
            else:
                candidate_ids = np.array(start_line_ids, dtype=int)

            # optionally filter out candidates that go to very near neighbors (min_distance_skip)
            if min_distance_skip > 0:
                keep = []
                for lid in candidate_ids:
                    _, (i_target, j_target) = (None, None)
                    i_target, j_target = lines_map[lid]
                    # compute index of target pin
                    # the mapping lines_map contains (i,j) already -> j is target
                    target_idx = lines_map[lid][1]
                    if dists[current_idx, target_idx] >= min_distance_skip:
                        keep.append(lid)
                if keep:
                    candidate_ids = np.array(keep, dtype=int)

        # score candidates
        scores = score_subset(candidate_ids)
        if scores.size == 0:
            fails += 1
            if fails >= 3:
                break
            continue
        best_local = np.argmax(scores)
        best_score = int(scores[best_local])
        best_line_id = int(candidate_ids[best_local])

        if best_score <= 0:
            fails += 1
            if fails >= 3:
                break
            continue

        # apply best line: update residual bytes by clearing covered bits
        chosen_mask = masks_uint8[best_line_id]
        # zero out bits where mask has 1 (we consider that drawing the line "covers" those pixels)
        res_bytes = np.bitwise_and(res_bytes, np.bitwise_not(chosen_mask))
        # update current_idx to target pin
        _, (from_i, to_j) = (None, lines_map[best_line_id])
        # lines_map[best_line_id] = (from_idx, to_idx)
        current_idx = lines_map[best_line_id][1]
        pull_order.append(current_idx)

        # termination quick check: if all residual bytes are zero -> target achieved
        if np.all(res_bytes == 0):
            logger.info("Residual exhausted: target matched.")
            break

    logger.info(f"Solver finished: iterations={iter_count}, pulls={len(pull_order)}")
    # cleanup
    gc.collect()
    return pull_order

# ---------- wrapper to run job ----------
def process_string_art_job(job_id: str, image_bytes: bytes, params: Dict[str,Any]):
    try:
        logger.info(f"[{job_id}] processing start")
        jobs_store[job_id]["status"] = JobStatus.PROCESSING
        jobs_store[job_id]["started_at"] = datetime.now().isoformat()

        # load image
        pil = Image.open(BytesIO(image_bytes))
        pil = pil.convert("RGB")
        img_np = np.array(pil).astype(np.float32) / 255.0

        # preprocess
        side_len = int(params.get("side_len", 400))
        binary_target, importance_map = preprocess_portrait(img_np, side_len=side_len, do_face_boost=True)

        H,W = binary_target.shape
        nails_count = int(params.get("nails_count", 200))
        nails = create_circle_nail_positions((H,W), nail_count=nails_count)
        nail_labels = create_nail_labels(nails_count)

        # solver params
        pull_amount = params.get("pull_amount")  # None or integer
        random_nails = params.get("random_nails", 120)  # candidate subset per iteration (tune)
        min_distance_skip = params.get("min_distance_skip", 2)

        # run solver
        pull_order = solver_precomputed_mask(binary_target, importance_map, nails, pull_amount=pull_amount, random_nails=random_nails, min_distance_skip=min_distance_skip)

        # render output image from pull_order -> draw antialiased lines on white canvas with black thread
        canvas = np.ones((H,W), dtype=np.float32)  # white background
        # draw lines
        for a,b in zip(pull_order, pull_order[1:]):
            rr, cc, val = line_aa(nails[a][0], nails[a][1], nails[b][0], nails[b][1])
            canvas[rr, cc] = np.clip(canvas[rr, cc] - val*0.95, 0.0, 1.0)

        # create base64 of final image
        image_b64 = np_to_base64_png(canvas, cmap='gray')

        # convert pull_order indices to labels
        labeled = []
        for idx in pull_order:
            if 0 <= idx < len(nail_labels):
                labeled.append(nail_labels[idx])
            else:
                labeled.append(str(idx))

        jobs_store[job_id].update({
            "status": JobStatus.COMPLETED,
            "completed_at": datetime.now().isoformat(),
            "results": {
                "image_base64": image_b64,
                "pull_order_indices": "-".join([str(i) for i in pull_order]),
                "pull_order_labels": "-".join(labeled),
                "total_pulls": len(pull_order),
                "nails_count": nails_count,
                "canvas_size": (H,W),
            },
            "metadata": {
                "params": params
            }
        })
        logger.info(f"[{job_id}] processing completed - pulls={len(pull_order)}")
    except Exception as e:
        logger.exception(f"[{job_id}] job failed: {e}")
        jobs_store[job_id].update({
            "status": JobStatus.FAILED,
            "error": str(e)
        })
    finally:
        gc.collect()

# ---------- API endpoints ----------
@app.post("/jobs")
async def submit_job(background_tasks: BackgroundTasks,
                     file: UploadFile = File(...),
                     side_len: int = 400,
                     nails_count: int = 200,
                     pull_amount: Optional[int] = None,
                     random_nails: Optional[int] = 120,
                     min_distance_skip: Optional[int] = 2):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")
    data = await file.read()
    job_id = str(uuid.uuid4())
    params = {
        "side_len": side_len,
        "nails_count": nails_count,
        "pull_amount": pull_amount,
        "random_nails": random_nails,
        "min_distance_skip": min_distance_skip
    }
    jobs_store[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        "filename": file.filename,
        "params": params
    }
    background_tasks.add_task(process_string_art_job, job_id, data, params)
    return {"job_id": job_id, "status": JobStatus.PENDING}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="job not found")
    return jobs_store[job_id]

@app.get("/")
async def root():
    return {"message": "StringArt API (Option B) running", "info": "POST /jobs with image to generate"}

# ---------- run locally ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("stringart_api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
