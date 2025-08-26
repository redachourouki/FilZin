import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.draw import line_aa, ellipse_perimeter
from math import atan2
from skimage.transform import resize
from time import time
import argparse

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

def find_best_nail_position(current_position, nails, str_pic, orig_pic, str_strength):

    best_cumulative_improvement = -99999
    best_nail_position = None
    best_nail_idx = None
    
    if args.random_nails != None:
        nail_ids = np.random.choice(range(len(nails)), size=args.random_nails, replace=False)
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

def create_art(nails, orig_pic, str_pic, str_strength, i_limit=None):

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
            print(f"Iteration {i}")
        
        if i_limit == None:
            if fails >= 3:
                break
        else:
            if i > i_limit:
                break

        idx, best_nail_position, best_cumulative_improvement = find_best_nail_position(current_position, nails,
                                                                                       str_pic, orig_pic, str_strength)

        if best_cumulative_improvement <= 0:
            fails += 1
            continue

        pull_order.append(idx)
        best_overlayed_line, rr, cc = get_aa_line(current_position, best_nail_position, str_strength, str_pic)
        str_pic[rr, cc] = best_overlayed_line

        current_position = best_nail_position
        iter_times.append(time() - start_iter)

    print(f"Time: {time() - start}")
    print(f"Avg iteration time: {np.mean(iter_times)}")
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


def pull_order_to_array_rgb(orders, canvas, nails, colors, strength):
    color_order_iterators = [iter(zip(order, order[1:])) for order in orders]
    for _ in range(len(orders[0]) - 1):
        # pull colors alternately
        for color_idx, iterator in enumerate(color_order_iterators):
            pull_start, pull_end = next(iterator)
            rr_aa, cc_aa, val_aa = line_aa(
                nails[pull_start][0], nails[pull_start][1],
                nails[pull_end][0], nails[pull_end][1]
            )

            val_aa_colored = np.zeros((val_aa.shape[0], len(colors)))
            for idx in range(len(val_aa)):
                val_aa_colored[idx] = np.full(len(colors), val_aa[idx])

            canvas[rr_aa, cc_aa] += colors[color_idx] * val_aa_colored * strength

            # rr, cc = line(
            #     nails[pull_start][0], nails[pull_start][1],
            #     nails[pull_end][0], nails[pull_end][1]
            # )
            # canvas[rr, cc] = colors[color_idx]
    return np.clip(canvas, a_min=0, a_max=1)

def process_single_contrast(orig_pic, nails, shape, contrast_label, image_dimens, args):
    """Process a single contrast level and return the result"""
    print(f"\n=== Processing {contrast_label} contrast ===")
    
    if args.wb:
        str_pic = init_canvas(shape, black=True)
        pull_order = create_art(nails, orig_pic, str_pic, 0.05, i_limit=args.pull_amount)
        blank = init_canvas(image_dimens, black=True)
    else:
        str_pic = init_canvas(shape, black=False)
        pull_order = create_art(nails, orig_pic, str_pic, -0.05, i_limit=args.pull_amount)
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
        args.export_strength if args.wb else -args.export_strength
    )
    
    return result, pull_order

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create String Art')
    parser.add_argument('-i', action="store", dest="input_file")
    parser.add_argument('-o', action="store", dest="output_file", default="output.png")
    parser.add_argument('-d', action="store", type=int, dest="side_len", default=300)
    parser.add_argument('-s', action="store", type=float, dest="export_strength", default=0.1)
    parser.add_argument('-l', action="store", type=int, dest="pull_amount", default=None)
    parser.add_argument('-r', action="store", type=int, dest="random_nails", default=None)
    parser.add_argument('-r1', action="store", type=float, dest="radius1_multiplier", default=1)
    parser.add_argument('-r2', action="store", type=float, dest="radius2_multiplier", default=1)
    parser.add_argument('-n', action="store", type=int, dest="nail_step", default=4)
    parser.add_argument('--wb', action="store_true")
    parser.add_argument('--rgb', action="store_true")
    parser.add_argument('--rect', action="store_true")

    args = parser.parse_args()

    LONG_SIDE = 300

    img = mpimg.imread(args.input_file)
    
    if np.any(img>100):
        img = img / 255
    
    if args.radius1_multiplier == 1 and args.radius2_multiplier == 1:
        img = largest_square(img)
        img = resize(img, (LONG_SIDE, LONG_SIDE))

    shape = ( len(img), len(img[0]) )

    if args.rect:
        nails = create_rectangle_nail_positions(shape, args.nail_step)
    else:
        nails = create_circle_nail_positions(shape, args.nail_step, args.radius1_multiplier, args.radius2_multiplier)

    print(f"Nails amount: {len(nails)}")
    
    if args.rgb:
        print("RGB mode not supported for multi-contrast generation. Use grayscale mode instead.")
        exit(1)
    else:
        # Convert to grayscale
        base_grayscale = rgb2gray(img) * 0.9
        
        # Define contrast levels
        contrast_levels = {
            'low': 0.5,    # Low contrast
            'mid': 0.75,    # Medium contrast (original)
            'high': 0.87    # High contrast
        }
        
        image_dimens = int(args.side_len * args.radius1_multiplier), int(args.side_len * args.radius2_multiplier)
        
        # Generate base filename without extension
        base_filename = args.output_file.rsplit('.', 1)[0] if '.' in args.output_file else args.output_file
        
        # Process each contrast level
        for contrast_name, contrast_factor in contrast_levels.items():
            # Apply contrast adjustment
            adjusted_grayscale = adjust_contrast(base_grayscale, contrast_factor)
            
            # Process with the existing algorithm (completely unchanged)
            result, pull_order = process_single_contrast(
                adjusted_grayscale, nails, shape, contrast_name, image_dimens, args
            )
            
            # Save result
            output_filename = f"{base_filename}_{contrast_name}.png"
            mpimg.imsave(output_filename, result, cmap=plt.get_cmap("gray"), vmin=0.0, vmax=1.0)
            
            print(f"Saved {contrast_name} contrast result to: {output_filename}")
            print(f"Thread pull order by nail index:\n{'-'.join([str(idx) for idx in pull_order])}")
        
        print(f"\n=== All contrast variations completed ===")
        print(f"Generated files:")
        for contrast_name in contrast_levels.keys():
            print(f"  - {base_filename}_{contrast_name}.png")