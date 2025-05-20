from PIL import Image
import os
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.transform import resize
import time
import numpy as np

def targetted_compression(input_path, output_path, target_size,error =0.5):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not in path")
    image = imread(input_path)
    if image.shape[-1] == 4: #not entirely sure what an alpha channel is but we get rid of it if it exists
        image = image[..., :3]

    gray = rgb2gray(image)
    gray_resized = resize(gray, (128, 128), anti_aliasing=True)
    img_uint8 = (gray_resized*255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8, mode='L') 

    for q in range(100, 0, -1):
        start = time.time()
        img_pil.save(output_path, format="JPEG", quality=q)
        size_kb = os.path.getsize(output_path)/1024 # in kb for comparison
        length = time.time() - start
        if abs(size_kb - target_size) <= error:
            print(f"Quality {q} gives size {size_kb:.2f} KB")
            print(f"Compression time: {length:.2f} seconds") #the alogrithm for f-string stuff
            return q, size_kb

    print("failed to match target size")
    return None, None

if __name__ == "__main__":
    targetted_compression("cat.png", "jpeg_cat_compressed.jpg", 5.08)

    
    



