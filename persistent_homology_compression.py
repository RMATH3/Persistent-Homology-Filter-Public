import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim
from gudhi import CubicalComplex, bottleneck_distance
from gudhi.wasserstein import wasserstein_distance
from datasets import load_dataset, concatenate_datasets, Value
import time
from scipy.ndimage import gaussian_filter

# ==== SETTINGS ====
IMAGE_SIZE = (128, 128)
PERCENTAGES = np.arange(0, 1.01, 0.01)
MAX_IMAGES = 100
PLOT_FOLDER = "plots"
os.makedirs(PLOT_FOLDER, exist_ok=True)
SAVE_ROOT = "results"


# ==== FUNCTIONS ====
def compute_ph_diagram(img):
    """Builds a cubical complex from an image and returns its persistence diagram
    for dimensions 0 and 1."""
    cc = CubicalComplex(top_dimensional_cells=img.astype(np.float32))
    cc.persistence()
    return np.vstack((
        cc.persistence_intervals_in_dimension(0),
        cc.persistence_intervals_in_dimension(1)
    ))

def compute_frequency_indices(fft_spectrum):
    """Returns all (i, j) index pairs for the FFT spectrum frequencies."""
    height, width = fft_spectrum.shape
    return [(i, j) for i in range(height) for j in range(width)]

def _conj_index(shape, idx):
    """Given FFT shape and one frequency index, compute its conjugate symmetric index."""
    H, W = shape
    i, j = idx
    return ((-i) % H, (-j) % W)

def retain_frequency(fft_spectrum, freq):
    """Keeps only one frequency (and its conjugate) in the FFT, reconstructs the image, and returns its persistence diagram."""
    H, W = fft_spectrum.shape
    mask = np.zeros_like(fft_spectrum, dtype=complex)
    ci = _conj_index((H, W), freq)
    mask[freq] = fft_spectrum[freq]
    mask[ci] = fft_spectrum[ci]
    inv_img = np.fft.ifft2(mask).real
    return compute_ph_diagram(inv_img)

def compress_fft(fft_spectrum, ranked, percentage):
    """Retains a percentage of the top-ranked frequencies (and their conjugates),
    reconstructs the image, and returns two versions:
    - raw for PH computations
    - normalized for SSIM comparisons."""
    keep_count = max(1, int(len(ranked) * percentage))
    H, W = fft_spectrum.shape
    mask = np.zeros_like(fft_spectrum, dtype=bool)
    
    for freq, _ in ranked[:keep_count]:
        ci = _conj_index((H, W), freq)
        mask[freq] = True
        mask[ci] = True
    
    mask[0, 0] = True  
    compressed = fft_spectrum * mask
    recon = np.fft.ifft2(compressed).real 
    
    # IMPORTANT return raw for PH, scaled copy for SSIM fir better metrics
    recon_for_ph = recon.copy()
    recon_for_ssim = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
    
    return recon_for_ph, recon_for_ssim
    
    
def rank_frequencies(full_pd, fft_spectrum, alpha=0.7):
    """
    Ranks frequencies by Wasserstein distance between full and single-frequency PH diagrams,
    weighted to prioritize low frequencies. Returns a sorted list of scores.
    """
    print("  Computing Wasserstein distances for all frequencies...")
    freqs = compute_frequency_indices(fft_spectrum)
    scores = {}

    for idx, f in enumerate(freqs):
        pd_freq = retain_frequency(fft_spectrum, f)
        dist = wasserstein_distance(full_pd, pd_freq, order=1.0)
        freq_norm = np.sqrt(f[0]**2 + f[1]**2) + 1e-8  
        weight = 1.0 / freq_norm  
        score = dist * weight
        scores[f] = score

        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(freqs)} frequencies...")

    # Sort by hybrid score (low = important)
    ranked = sorted(scores.items(), key=lambda x: x[1])
    return ranked


def targetted_compression_array(img_array, output_path, target_size_kb, error=0.1):
    """
    Compresses an image array to JPEG format,
    iteratively adjusting quality until the file size matches a target..
    """
    img_uint8 = (img_array*255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8, mode='L')
    
    for q in range(100, 0, -1):
        start = time.time()
        img_pil.save(output_path, format="JPEG", quality=q)
        size_kb = os.path.getsize(output_path) / 1024
        elapsed = time.time() - start
        
        if abs(size_kb - target_size_kb) <= error:
            return q, size_kb
    
    return None, None

def betti_numbers(pd):
    """Compute Betti numbers (number of features in each dimension)."""
    bettis = [len(pd[pd[:, 0] == dim]) for dim in [0, 1]]  
    return np.array(bettis)

def betti_diff(pd1, pd2):
    """Returns the absolute difference in Betti numbers between two persistence diagrams."""
    return np.sum(np.abs(betti_numbers(pd1) - betti_numbers(pd2)))

# ==== LOAD DATASET ====
print("Loading datasets...")
tiny = load_dataset("Maysee/tiny-imagenet", split="train")
tiny = tiny.cast_column("label", Value("int64"))

cifar10 = load_dataset("cifar10", split="train")
cifar10 = cifar10.cast_column("label", Value("int64"))
cifar10 = cifar10.rename_column("img", "image")

stl10 = load_dataset("jxie/stl10", split="train")
stl10 = stl10.cast_column("label", Value("int64"))

combined = concatenate_datasets([tiny, cifar10, stl10])
dataset = combined.shuffle(seed=42)

print(tiny.column_names)
print(cifar10.column_names)

# ==== METRICS STORAGE ====
mse_acc_ph = {p: [] for p in PERCENTAGES}
ssim_acc_ph = {p: [] for p in PERCENTAGES}
size_acc_ph = {p: [] for p in PERCENTAGES}

mse_acc_jpeg = {p: [] for p in PERCENTAGES}
ssim_acc_jpeg = {p: [] for p in PERCENTAGES}
size_acc_jpeg = {p: [] for p in PERCENTAGES}

wasserstein_acc_ph = {p: [] for p in PERCENTAGES}
wasserstein_acc_jpeg = {p: [] for p in PERCENTAGES}

bottleneck_acc_ph = {p: [] for p in PERCENTAGES}
bottleneck_acc_jpeg = {p: [] for p in PERCENTAGES}

betti_acc_ph = {p: [] for p in PERCENTAGES}
betti_acc_jpeg = {p: [] for p in PERCENTAGES}

# ==== MAIN LOOP ====
start_time = time.time()
print(f"Starting processing of up to {MAX_IMAGES} images...")
count = 0
for example in dataset:
    if count >= MAX_IMAGES:
        print("Reached image limit. Stopping.")
        break

    img_start_time = time.time()
    print(f"\nProcessing image {count+1}/{MAX_IMAGES}...")
    
    img = example["image"].resize(IMAGE_SIZE)
    img_gray = np.array(img.convert("L")) / 255.0
    
    print("  Computing full persistence diagram...")
    full_pd = compute_ph_diagram(img_gray)
    
    print("  Performing FFT on image...")
    fft_img = np.fft.fft2(img_gray)
    ranked = rank_frequencies(full_pd, fft_img)
    img_dir = os.path.join(SAVE_ROOT, f"image_{count+1:03d}")
    os.makedirs(img_dir, exist_ok=True)
    # Save original image
    original_path = os.path.join(img_dir, "original.png")
    img.save(original_path)

    for p in PERCENTAGES:
        print(f"  → Retaining {p*100:.0f}% of frequencies...")
        # ===== PH-BASED COMPRESSION =====
        recon_ph, recon_ssim = compress_fft(fft_img, ranked, p)
        mse_val_ph = mse(img_gray, recon_ssim)      # SSIM/MSE use normalized version
        ssim_val_ph = ssim(img_gray, recon_ssim, data_range=1.0)
        # --- Apply light smoothing ONLY for PH ---
        recon_ph_filtered = gaussian_filter(recon_ph, sigma=1)

        # Use filtered version for topological metrics not the others
        pd_recon_ph = compute_ph_diagram(recon_ph_filtered)
        ph_path = os.path.join(img_dir, f"ph_{int(p*100)}.png")
        Image.fromarray((recon_ph_filtered * 255).astype(np.uint8), mode="L").save(ph_path)
        temp_path_ph = os.path.join(PLOT_FOLDER, f"tmp_ph_{p}.jpg")
        ph_path = os.path.join(img_dir, f"ph_{int(p*100)}.png")
        Image.fromarray((recon_ph_filtered * 255).astype(np.uint8), mode="L").save(ph_path)
        sz_kb_ph = os.path.getsize(ph_path) / 1024

        
        mse_acc_ph[p].append(mse_val_ph)
        ssim_acc_ph[p].append(ssim_val_ph)
        size_acc_ph[p].append(sz_kb_ph)
        
        # ===== JPEG-ONLY COMPRESSION =====
        temp_path_jpeg = os.path.join(PLOT_FOLDER, f"tmp_jpeg_{p}.jpg")
        
        jpeg_quality = int(100 * p)
        jpeg_quality = max(5, min(95, jpeg_quality))

        # Convert grayscale numpy array to PIL image
        img_uint8 = (img_gray * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode="L")
        #save image
        jpeg_path = os.path.join(img_dir, f"jpeg_{int(p*100)}.jpg")
        img_pil.save(jpeg_path, format="JPEG", quality=jpeg_quality, subsampling=2)
        jpeg_img = np.array(Image.open(jpeg_path).convert("L")) / 255.0
        mse_val_jpeg = mse(img_gray, jpeg_img)
        ssim_val_jpeg = ssim(img_gray, jpeg_img, data_range=1.0)
        sz_kb_jpeg = os.path.getsize(jpeg_path) / 1024
        mse_acc_jpeg[p].append(mse_val_jpeg)
        ssim_acc_jpeg[p].append(ssim_val_jpeg)
        size_acc_jpeg[p].append(sz_kb_jpeg)
        
        pd_recon_ph = compute_ph_diagram(recon_ph)
        pd_jpeg = compute_ph_diagram(jpeg_img)
        
        # Wasserstein
        ws_ph = wasserstein_distance(full_pd, pd_recon_ph, order=1.0)
        ws_jpeg = wasserstein_distance(full_pd, pd_jpeg, order=1.0)
        wasserstein_acc_ph[p].append(ws_ph)
        wasserstein_acc_jpeg[p].append(ws_jpeg)
        
        # Bottleneck
        bn_ph = bottleneck_distance(full_pd, pd_recon_ph)
        bn_jpeg = bottleneck_distance(full_pd, pd_jpeg)
        bottleneck_acc_ph[p].append(bn_ph)
        bottleneck_acc_jpeg[p].append(bn_jpeg)
        
        # Betti number difference
        bd_ph = betti_diff(full_pd, pd_recon_ph)
        bd_jpeg = betti_diff(full_pd, pd_jpeg)
        betti_acc_ph[p].append(bd_ph)
        betti_acc_jpeg[p].append(bd_jpeg)
        
    
    img_elapsed = time.time() - img_start_time
    print(f"  Image processed in {img_elapsed:.2f} seconds.")
    count += 1

print("\nAll images processed. Generating graphs...")

total_elapsed = time.time() - start_time
avg_time = total_elapsed / count if count > 0 else 0

print(f"\nTotal run time: {int(total_elapsed // 3600)}h {int((total_elapsed % 3600) // 60)}m {int(total_elapsed % 60)}s")
print(f"Average time per image: {int(avg_time // 3600)}h {int((avg_time % 3600) // 60)}m {int(avg_time % 60)}s")

# ==== COMPARISON PLOTTING ====
metrics_ph = {
    "MSE": mse_acc_ph,
    "SSIM": ssim_acc_ph,
    "JPEG Size KB": size_acc_ph,
    "Wasserstein": wasserstein_acc_ph,
    "Bottleneck": bottleneck_acc_ph,
    "Betti": betti_acc_ph
}

metrics_jpeg = {
    "MSE": mse_acc_jpeg,
    "SSIM": ssim_acc_jpeg,
    "JPEG Size KB": size_acc_jpeg,
    "Wasserstein": wasserstein_acc_jpeg,
    "Bottleneck": bottleneck_acc_jpeg,
    "Betti": betti_acc_jpeg
}
# All PH curves = blue, all JPEG curves = orange
ph_color = "blue"
jpeg_color = "orange"

markers = {"PH": "o", "JPEG": "s"}
linestyles = {"PH": "-", "JPEG": "--"}
x = [p * 100 for p in PERCENTAGES]
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

for i, metric_name in enumerate(metrics_ph.keys()):
    # PH metrics
    means_ph = [np.mean(metrics_ph[metric_name][p]) for p in PERCENTAGES]
    axs[i].plot(x, means_ph, marker=markers["PH"], linestyle=linestyles["PH"],
                color=ph_color, label="PH Compression")
    
    # JPEG metrics
    means_jpeg = [np.mean(metrics_jpeg[metric_name][p]) for p in PERCENTAGES]
    axs[i].plot(x, means_jpeg, marker=markers["JPEG"], linestyle=linestyles["JPEG"],
                color=jpeg_color, label="JPEG Compression")
    axs[i].set_xlabel("Percentage of Frequencies Kept (%)", fontsize=11)
    axs[i].set_ylabel(f"Mean {metric_name}", fontsize=11)
    axs[i].set_title(f"{metric_name} Comparison", fontsize=13)
    axs[i].grid(True, which="both", linestyle="--", linewidth=0.6)
    axs[i].legend(loc="best", fontsize=10)
fig.suptitle("PH vs JPEG Compression Metrics Across Frequencies", fontsize=16, y=1.02)
plt.tight_layout()
output_path = os.path.join(PLOT_FOLDER, "ph_vs_jpeg_comparison_full_labels.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plots saved to {output_path}")
