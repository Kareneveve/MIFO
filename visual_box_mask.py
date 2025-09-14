import numpy as np
from PIL import Image
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="args for visualizing box")

    parser.add_argument('--top_left', 
                    type=lambda x: tuple(map(float, x)), 
                    nargs=2, 
                    default=(0.25, 0.3), 
                    help='Upper left corner coordinate')
    parser.add_argument('--bottom_right', 
                    type=lambda x: tuple(map(float, x)), 
                    nargs=2, 
                    default=(0.25, 0.3), 
                    help='Lower right corner coordinates')

    args = parser.parse_args()
    return args

def visual_box_mask(top_left, bottom_right,image_width = 512, image_height = 512):
    if not (0 <= top_left[0] <= 1 and 0 <= top_left[1] <= 1 and 
            0 <= bottom_right[0] <= 1 and 0 <= bottom_right[1] <= 1):
        raise ValueError("The coordinate value must be between 0 and 1.")
        
    if top_left[0] >= bottom_right[0] or top_left[1] >= bottom_right[1]:
        raise ValueError("The upper left corner coordinate must be smaller than the lower right corner coordinate.")
    
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    x1 = int(top_left[0] * image_width)
    y1 = int(top_left[1] * image_height)
    x2 = int(bottom_right[0] * image_width)
    y2 = int(bottom_right[1] * image_height)
    
    mask[y1:y2, x1:x2] = 1
    
    mask_image = Image.fromarray(mask * 255)  
    
    return mask_image


if __name__ == "__main__":
    arg = get_args()
    mask_image = visual_box_mask(top_left = arg.top_left, bottom_right = arg.bottom_right)
    mask_image.save("mask.png")