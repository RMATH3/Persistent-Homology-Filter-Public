import numpy as np
import matplotlib.pyplot as plt
from gudhi import CubicalComplex
from gudhi.wasserstein import wasserstein_distance
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
import os

def compute_ph_diagram(img):
    img = np.array(img, dtype=np.float32)
    cc = CubicalComplex(top_dimensional_cells=img)
    cc.persistence()
    return np.vstack((
        cc.persistence_intervals_in_dimension(0),
        cc.persistence_intervals_in_dimension(1)
    ))


'''def keep_freqs_via_ph(original_image, initial_epsilon=11.040, max_freqs=None, decay_rate=1):
    original_ph = compute_ph_diagram(original_image)
    fft_image = np.fft.fft2(original_image)
    kept_mask = np.zeros_like(fft_image, dtype=bool)

    height, width = fft_image.shape
    freqs = [(u, v) for u in range(height) for v in range(width)]

    # Sort frequencies by magnitude, descending — more likely relevant first
    freqs.sort(key=lambda pos: np.abs(fft_image[pos]), reverse=True)

    kept_count = 0
    evaluated = 0
    epsilon = initial_epsilon
    max_freqs = max_freqs or len(freqs)

    for (u, v) in freqs:
        if evaluated >= max_freqs:
            print(f"Stopping early at {max_freqs} frequencies evaluated.")
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

    return fft_image * kept_mask'''

def denoise_via_ph(original_image, initial_epsilon=11.0, max_freqs=None, decay_rate=1):
    original_ph = compute_ph_diagram(original_image)
    fft_image = np.fft.fft2(original_image)
    denoised_fft = fft_image.copy()

    height, width = fft_image.shape
    freqs = [(u, v) for u in range(height) for v in range(width)]

    # Sort by magnitude ascending — more likely to remove small/noisy frequencies first
    freqs.sort(key=lambda pos: np.abs(fft_image[pos]))

    evaluated = 0
    removed_count = 0
    epsilon = initial_epsilon
    max_freqs = max_freqs or len(freqs)

    for (u, v) in freqs:
        if evaluated >= max_freqs:
            print(f"Stopping early at {max_freqs} frequencies evaluated.")
            break

        original_val = denoised_fft[u, v]
        denoised_fft[u, v] = 0  # Try removing

        inv = np.fft.ifft2(denoised_fft).real
        freq_ph = compute_ph_diagram(inv)

        try:
            dist = wasserstein_distance(original_ph, freq_ph, order=1.)
        except Exception:
            dist = float("inf")

        if dist > epsilon:
            # Too important — restore it
            denoised_fft[u, v] = original_val
        else:
            removed_count += 1

        evaluated += 1
        if evaluated % 100 == 0:
            print(f"Evaluated {evaluated}, removed {removed_count}, ε = {epsilon:.3f}")
            epsilon *= decay_rate  # gradually loosen

    return denoised_fft


filename = "cat.png"  # replace this with your file path
if not os.path.exists(filename):
    raise FileNotFoundError(f"Could not find image: {filename}")

image = imread(filename)
if image.shape[-1] == 4:  # RGBA to RGB
    image = image[..., :3]

gray = rgb2gray(image)
gray_resized = resize(gray, (128, 128), anti_aliasing=True)

# 2. Add Gaussian noise
np.random.seed(0)
sigma = 0.1
noisy_image = gray_resized + sigma * np.random.randn(*gray_resized.shape)


# === Run compression ===
kept_fft = denoise_via_ph(
    noisy_image,
    initial_epsilon=28,
    max_freqs=None,       # Increase if you want more precision
    decay_rate=1.05       # Relax threshold slowly
)

# === Reconstruct image ===
reconstructed = np.fft.ifft2(kept_fft).real

# === Display results ===
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(noisy_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Kept Frequencies (log abs)")
plt.imshow(np.log1p(np.abs(np.fft.fftshift(kept_fft))), cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Reconstructed (PH-Preserved)")
plt.imshow(reconstructed, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()