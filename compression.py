import numpy as np
import matplotlib.pyplot as plt
from gudhi import CubicalComplex
from gudhi.wasserstein import wasserstein_distance
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.transform import resize
import os
import time
from skimage import img_as_ubyte


def compute_ph_diagram(img):
    img = np.array(img, dtype=np.float32)
    cc = CubicalComplex(top_dimensional_cells=img)
    cc.persistence()
    return np.vstack((
        cc.persistence_intervals_in_dimension(0),
        cc.persistence_intervals_in_dimension(1)
    ))
#epsilon of 11.040 found via trial and error
def keep_freqs(original_image, initial_epsilon=11.040, max_freqs=None, decay_rate=1):
    original_ph = compute_ph_diagram(original_image)
    fft_image = np.fft.fft2(original_image)
    kept_mask = np.zeros_like(fft_image, dtype=bool)

    height, width = fft_image.shape
    freqs = [(u, v) for u in range(height) for v in range(width)]

    # sort by frequency
    freqs.sort(key=lambda pos: np.abs(fft_image[pos]), reverse=True)

    kept_count = 0
    evaluated = 0
    epsilon = initial_epsilon
    max_freqs = max_freqs or len(freqs)

    for (u, v) in freqs:
        if evaluated >= max_freqs:
            print(f"stopped early at {max_freqs}")
            break

        single_freq = np.zeros_like(fft_image, dtype=complex)
        single_freq[u, v] = fft_image[u, v]

        inv = np.fft.ifft2(single_freq).real
        freq_ph = compute_ph_diagram(inv)

        try:
            dist = wasserstein_distance(original_ph, freq_ph, order=1.)
        except Exception:
            dist = float("inf")

        if dist < epsilon:
            kept_mask[u, v] = True
            kept_count += 1

        evaluated += 1
        if evaluated % 100 == 0:
            print(f"Evaluated {evaluated}, kept {kept_count}, ε = {epsilon:.3f}")
            epsilon *= decay_rate  # gradually loosen

    return fft_image * kept_mask

#loading/preprocessing image
print(os.listdir("."))
filename = "cat.png"
if not os.path.exists(filename):
    raise FileNotFoundError(f"{filename} not in path")

image = imread(filename)
if image.shape[-1] == 4: #not entirely sure what an alpha channel is but we get rid of it if it exists
    image = image[..., :3]

gray = rgb2gray(image)
gray_resized = resize(gray, (128, 128), anti_aliasing=True)

# === Run compression ===
start = time.time()
kept_fft = keep_freqs(
    gray_resized,
    initial_epsilon=11.040,
    max_freqs=None,      
    decay_rate=1      
)
compression_time = time.time() - start
print(f"Compression time: {compression_time:.2f} seconds")

reconstructed = np.fft.ifft2(kept_fft).real
clipped = np.clip(reconstructed, 0, 1)

uint8_img = img_as_ubyte(clipped)
imsave("ph_compressed_cat.png", uint8_img)
ph_size = os.path.getsize("ph_compressed_cat.png")
print(f"PH size: {ph_size/1024:.2f} kb") #file size in kb

# obtained help for matplot
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image", fontsize=16)
plt.imshow(gray_resized, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Kept Frequencies (log abs)",fontsize=16)
plt.imshow(np.log1p(np.abs(np.fft.fftshift(kept_fft))), cmap="gray") #shift frequencies to center
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Reconstructed (PH-Preserved)",fontsize=16)
plt.imshow(reconstructed, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()