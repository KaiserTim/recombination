import os
import numpy as np
import zipfile
import shutil

from tqdm import tqdm
from PIL import Image


npz_file = "/home/shared/generative_models/recombination/raw_samples/in64/visual-vae/50000_random_classes_m0.0_v0.0.npz"
tmp_dir = "/tmp/npz_to_images"  # Temporary directory for images

# Path to save the final zip file
zip_file = "/home/shared/generative_models/recombination/raw_samples/in64/visual-vae/50000_random_classes_m0.0_v0.0.zip"

data = np.load(npz_file)
images = data["arr0"]
assert images.ndim == 4 and images.shape[-1] == 3, "Expected images array of shape (N, H, W, 3)"

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

print("Converting numpy images to PNG files...")
for i, img in tqdm(enumerate(images), total=len(images)):
    img = Image.fromarray(img.astype('uint8'))  # Convert image array to PIL Image
    img.save(os.path.join(tmp_dir, f"{i:05d}.png"))  # Save with unique name

print("Adding images to zip file...")
with zipfile.ZipFile(zip_file, "w") as zf:
    for root, _, files in os.walk(tmp_dir):
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            zf.write(file_path, arcname=file)  # Add file to zip with its relative name

shutil.rmtree(tmp_dir)

print(f"Zip file created at: {zip_file}")
