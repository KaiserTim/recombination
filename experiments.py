import os
import click
import torch
import numpy as np
import faiss
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sys
import glob
import dnnlib

from matplotlib import colormaps as cm
from torchvision.utils import make_grid
from PIL import Image, ImageDraw
from scipy.stats import entropy
from tqdm import tqdm



# ----------- Helper Functions -----------

def load_embeddings(path, n=-1):
    """Load .pt embeddings to the desired device (GPU by default). Limit size to n.
    
    Args:
        path (str): Path to the .pt file containing embeddings.
        n (int, optional): Maximum number of embeddings to load. Defaults to -1 (all).
    
    Returns:
        torch.Tensor: Loaded embeddings tensor, limited to n items if specified.
    """
    embeds = torch.load(path)
    assert len(embeds) >= n, f"n is bigger than the saved embeddings ({n} > {len(embeds)}). Please adjust n_train or n_gen."
    return embeds[:n]


def prepare_embeddings(embeddings):
    """Flatten patchwise tensor from (n, 256, D) → (n*256, D).
    
    Args:
        embeddings (torch.Tensor): Input tensor of shape (n, 256, D).
    
    Returns:
        tuple: Contains:
            - torch.Tensor: Flattened tensor of shape (n*256, D)
            - int: Original number of images n
    """
    N, P, D = embeddings.shape
    embeddings_np = embeddings.view(-1, D).detach().cpu().numpy().astype('float32')

    # Normalize features
    # train_np = train_np / np.linalg.norm(train_np, axis=-1, keepdims=True)
    faiss.normalize_L2(embeddings_np)

    return embeddings_np, N


# ----------- A. Nearest Neighbor Patch Matches -----------

def create_index(path_train, n_train, save_dir, index_func, chunk_size=1024):
    """Creates a FAISS index from training embeddings in chunks and saves to the specified directory.

    This function processes a specified number of training embeddings, divides them into smaller
    chunks, and computes a FAISS index for each chunk. The resulting index chunks are saved locally
    to the specified directory, with proper handling for GPU-based computations.

    Args:
        path_train (str): Path to the file containing training embeddings.
        n_train (int): Number of training embeddings to load.
        save_dir (str): Directory where the FAISS index chunks will be saved.
        chunk_size (int, optional): Number of embeddings to include in each chunk while creating 
            the FAISS index. Defaults to 1024.

    Returns:
        None
    """
    chunk_dir = os.path.join(save_dir, "faiss_index")
    os.makedirs(chunk_dir, exist_ok=True)

    train_embeds = None
    pbar = None
    for i in range(0, n_train, chunk_size):
        chunk_path = os.path.join(chunk_dir, f"chunk_{i:07d}.bin")
        if os.path.isfile(chunk_path):
            continue
        if pbar is None:  # Initialize the progress bar only if needed
            train_embeds = load_embeddings(path_train, n_train)
            print(f"Loaded {train_embeds.shape[0]} training image embeddings with {train_embeds.shape[1]} patches each.")
            pbar = tqdm(range(0, n_train, chunk_size), desc="", dynamic_ncols=True, file=sys.stdout)
        chunk = train_embeds[i:i + chunk_size].to('cuda')  # Slice the chunk
        chunk_index = compute_faiss_index(chunk, index_func, use_gpu=True)
        faiss.write_index(faiss.index_gpu_to_cpu(chunk_index), chunk_path)
        if pbar is not None:
            pbar.set_description(f"Creating new FAISS index in chunks of size {chunk_size}")
        pbar.update(n=1)
    if pbar is not None:
        pbar.close()
        print(f"Saved FAISS index chunks to {chunk_dir}")


def compute_faiss_index(data, index_func, use_gpu, batch_size=1024):
    """
    Create a FAISS index for nearest neighbor search.
    Inputs:
        data: Embeddings of image patches, (n_imgs, n_patches, D)
        use_gpu: Whether to use GPU for FAISS index
         metric: Distance metric to use ('l2' or 'cosine')
        batch_size: Batch size for FAISS search (higher values will increase search speed but require more memory)
    Returns:
        all_indices: Indices of k-nearest neighbors from the training data for each patch, (n_gen, n_patches, k)
        all_distances: The corresponding distances, (n_gen, n_patches, k)
    """
    n_gen, n_patches, _ = data.shape
    train_np, n_imgs = prepare_embeddings(data)  # (n_imgs * n_patches, D)
    embed_dim = train_np.shape[1]

    def add_to_faiss_index(index, data, batch_size, use_gpu):
        """Helper function to add batches of data to the FAISS index with a progress bar."""
        # pbar = tqdm(total=len(data), desc=f"Building FAISS index with {n_gen} images and {n_patches} patches each.", file=sys.stdout, dynamic_ncols=True, leave=True)
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            index.add(batch)
            # if use_gpu:  # Display GPU memory usage if applicable
            #     allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # In GB
            #     reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # In GB
                # pbar.set_postfix({"GPU Alloc (GB)": f"{allocated:.2f}", "GPU Resrv (GB)": f"{reserved:.2f}"})
            # pbar.update(len(batch))
        # pbar.close()

    # Build FAISS index
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = index_func(embed_dim)
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        index = index_func(embed_dim)

    add_to_faiss_index(index, train_np, batch_size, use_gpu)
    assert index.ntotal == n_imgs * n_patches, "Index size does not match the number of training patches."
    return index


def compute_patchwise_nearest_neighbors(
        gen_embeds,  # Generated embeddings, shape (n_gen, n_patches, D)
        chunk_dir,  # Directory containing the FAISS index chunks
        n_train,  # Number of total training images to use
        index_func,  # FAISS index function, either faiss.IndexFlatL2 or faiss.IndexFlatIP
        k=1,  # Number of nearest neighbors to return
        use_gpu=True,  # Use GPU if available
        batch_size=16,  # Batch size for FAISS search
        metric='l2',  # Distance metric used ('l2' or 'cosine')
):
    """
    Compute the nearest neighbors of each generated patch using FAISS index chunks.
    Instead of loading the entire FAISS index into memory, process one index chunk at a time.

    Args:
        gen_embeds: Embeddings of generated patches, shape (n_gen, n_patches, D).
        chunk_dir: Path to the directory containing FAISS index chunks.
        n_train: Number of training images to use.
        index_func: FAISS index function (IndexFlatL2 or IndexFlatIP).
        k: The number of nearest neighbors to return.
        use_gpu: Whether to use GPU for FAISS index processing.
        batch_size: Batch size for FAISS search.
        metric: Distance metric ('l2' or 'cosine').

    Returns:
        all_indices: Indices of k-nearest neighbors for each generated patch, shape (n_gen, n_patches, k).
        all_distances: Corresponding distances of k-nearest neighbors, shape (n_gen, n_patches, k).
    """

    # Flatten generated embeddings
    n_gen, n_patches, dim = gen_embeds.shape
    print(f"Performing NN search for {n_gen} images with {n_patches} patches each...")
    gen_np, _ = prepare_embeddings(gen_embeds)  # Shape: (n_gen * n_patches, embed_dim)

    # Prepare overall results
    all_indices = np.full((len(gen_np), k), -1, dtype=np.int32)  # Initialize to -1
    if metric == 'l2':
        # For L2, we want to minimize distance, so initialize to infinity
        all_distances = np.full((len(gen_np), k), np.inf, dtype=np.float32) 
    else:
        # For cosine, we want to maximize similarity, so initialize to negative infinity
        all_distances = np.full((len(gen_np), k), -np.inf, dtype=np.float32) 

    # Iterate through FAISS index chunks
    chunk_paths = sorted(glob.glob(f"{chunk_dir}/chunk_*.bin"))  # Sorted to ensure correct order
    chunk_paths = [path for path in chunk_paths if int(path.split('_')[-1].split('.')[0]) < n_train]
    print(f"Using {len(chunk_paths)} chunks from {n_train} training images from directory: {chunk_dir}")

    pbar = tqdm(enumerate(chunk_paths), total=len(chunk_paths), desc="", dynamic_ncols=True, file=sys.stdout)
    for chunk_idx, chunk_path in pbar:
        pbar.set_description(f"Chunk {chunk_idx + 1}/{len(chunk_paths)} NN search")
        # Load and potentially truncate the index chunk
        index = faiss.read_index(chunk_path)
        chunk_start_patch = int(chunk_path.split('_')[-1].split('.')[0]) * n_patches
        if chunk_idx == len(chunk_paths) - 1:  # Trim the last chunk if needed
            total_allowed = n_train * n_patches
            current_offset = chunk_start_patch
            remaining_allowed = total_allowed - current_offset

            if remaining_allowed < index.ntotal:
                # Truncate index to only keep the needed amount
                new_index = index_func(index.d)
                for i in range(remaining_allowed):  # Not efficient but fast enough when chunks are small
                    new_index.add(np.expand_dims(index.reconstruct(i), axis=0))
                index = new_index

        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        # Perform batched nearest-neighbor search on this chunk
        for i in range(0, len(gen_np), batch_size):
            batch = gen_np[i:i + batch_size]  # Batch of query patches
            chunk_distances, chunk_indices = index.search(batch, k)  # Perform search on this chunk

            # Adjust indices to account for chunk offset
            chunk_indices += chunk_start_patch

            # Update overall results
            for j in range(len(batch)):
                # Combine neighbors across chunks
                combined_distances = np.concatenate((all_distances[i + j], chunk_distances[j]))
                combined_indices = np.concatenate((all_indices[i + j], chunk_indices[j]))
                
                # Sort based on the metric
                if metric == 'l2':
                    # For L2, we want the smallest distances (ascending order)
                    sorted_indices = np.argsort(combined_distances)[:k]  
                else:
                    # For cosine similarity, we want the largest values (descending order)
                    sorted_indices = np.argsort(-combined_distances)[:k]  
                
                all_distances[i + j] = combined_distances[sorted_indices]
                all_indices[i + j] = combined_indices[sorted_indices]

    # Reshape results back to match original input shape
    all_indices = all_indices.reshape(n_gen, n_patches, k)
    all_distances = all_distances.reshape(n_gen, n_patches, k)

    return all_indices, all_distances


# ----------- C. Patch Origin Metrics -----------

def compute_patch_origin_histogram(nn_indices, n_train):
    """Count how many of the patches came from each training image.

    Args:
        nn_indices (numpy.ndarray): Indices of k-nearest neighbors from the training data for each patch, shape (n_gen, n_patches, k).
        n_train (int): Number of training images.

    Returns:
        numpy.ndarray: Histogram counting patch origins per training image.
    """
    n_patches = nn_indices.shape[1]
    patch_indices = nn_indices.flatten()  # (n_gen * n_patches * k,)
    assert patch_indices.max() <= n_train * n_patches - 1, f"Max patch index: {patch_indices.max()} exceeds expected max patch index: {n_train * n_patches - 1}."
    image_ids = patch_indices // n_patches  # Maps training patch indices to training image indices
    print(f"NN Indices Shape: {nn_indices.shape}")
    hist = np.bincount(image_ids, minlength=n_train)
    return hist


def compute_patch_origin_entropy(hist):
    """Computes the entropy of the patch origin histogram.
    
    Args:
        hist (numpy.ndarray): Histogram of patch origins.
    
    Returns:
        float: Entropy value of the patch origin distribution.
    """
    prob = hist / np.sum(hist)
    return entropy(prob, base=2)


def compute_unique_source_count(hist):
    """Computes the number of unique source images used.

    Args:
        hist (numpy.ndarray): Histogram of patch origins.

    Returns:
        int: Number of unique source images with non-zero patch counts.
    """
    return np.count_nonzero(hist)


# ----------- E. Visualization -----------

def create_colored_sources_plot(image, indices, n_train, patch_colors=None, alpha=0.25):
    """
    Creates a color-coded patch source map overlay for a generated image.

    Args:
        image (PIL.Image): The original generated image
        indices (np.ndarray): Nearest neighbor indices, shape (n_patches,)
        n_train (int): Total number of training images
        patch_colors (dict, optional): Dictionary mapping (i,j) patch coordinates to colors
                                      from the heatmap. If None, uses random colors.
        alpha (float): Opacity for the overlay colors

    Returns:
        PIL.Image: Image with colored source overlay
        np.ndarray: Image source IDs
    """
    H, W = image.size
    # Get patch-level source IDs
    patch_sources = indices.squeeze()  # Shape: (n_patches,)
    n_patches = patch_sources.shape[0]
    patch_size = int(H / n_patches ** 0.5)  # Figure out patch dimensions
    image_sources = patch_sources // n_patches  # Map patch IDs to image IDs

    # Create overlay
    image_rgba = image.convert("RGBA")  # Ensure image is in RGBA format for blending
    overlay = Image.new("RGBA", image_rgba.size)  # Create a blank overlay
    draw = ImageDraw.Draw(overlay, "RGBA")

    sqrt_n_patches = int(n_patches ** 0.5)
    for i in range(sqrt_n_patches):
        for j in range(sqrt_n_patches):
            patch_idx = i * sqrt_n_patches + j
            
            if patch_colors and (i, j) in patch_colors:
                # Use colors from heatmap if provided
                color_rgb = patch_colors[(i, j)]
                color = tuple(int(c * 255) for c in color_rgb) + (int(255 * alpha),)
            else:
                # Fall back to random colors based on source IDs
                normed_source = image_sources[patch_idx] / n_train
                color_rgb = cm['hsv'](normed_source)[:3]
                color = tuple(int(c * 255) for c in color_rgb) + (int(255 * alpha),)
            
            x0, y0 = j * patch_size, i * patch_size
            x1, y1 = x0 + patch_size, y0 + patch_size
            draw.rectangle([x0, y0, x1, y1], fill=color)

    # Blend the overlay with the original image using alpha compositing
    image_with_overlay = Image.alpha_composite(image_rgba, overlay)

    return image_with_overlay, image_sources


def create_patch_reconstruction_grid(indices, train_dataset, H, W):
    """
    Creates a grid of nearest neighbor patches from the training dataset.

    Args:
        indices (np.ndarray): Nearest neighbor indices, shape (n_patches,)
        image_index (int): Index of the generated image
        train_dataset: A dataset of training images
        H, W (int): Height and width of the original image

    Returns:
        np.ndarray: Grid of nearest neighbor patches
    """
    patch_sources = indices.squeeze()  # Shape: (n_patches,)
    n_patches = patch_sources.shape[0]
    sqrt_n_patches = int(n_patches ** 0.5)
    patch_size = H // sqrt_n_patches
    image_sources = patch_sources // n_patches  # Get corresponding training image IDs

    # Extract patches from training images
    matched_patches = []
    for i, train_img_idx in enumerate(image_sources[:n_patches]):  # Limit to grid size
        train_image = train_dataset[train_img_idx][0].transpose(1, 2, 0)  # Load training image and convert
        patch_y = (i // sqrt_n_patches) * patch_size
        patch_x = (i % sqrt_n_patches) * patch_size
        patch = train_image[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size, :]
        matched_patches.append(torch.from_numpy(patch).permute(2, 0, 1))  # Convert back to tensor format

    # Create a grid of matched patches
    patch_grid = make_grid(matched_patches, nrow=sqrt_n_patches, padding=0).numpy()  # (C, H, W)

    return patch_grid.transpose(1, 2, 0)  # Convert to (H, W, C) for display


def create_distance_heatmap(distances, image_index, H, W, metric='l2'):
    """
    Creates a patch-based heatmap showing the distance/similarity of each patch to its nearest neighbor.

    Args:
        distances (np.ndarray): Distances/similarities to nearest neighbors, shape (n_gen, n_patches, k)
        image_index (int): Index of the generated image
        H, W (int): Height and width of the original image
        metric (str): Metric used ('l2' or 'cosine')

    Returns:
        tuple:
            - np.ndarray: Patch-based heatmap with dimensions (H, W, 3)
            - dict: Metadata including min/max values and patch colors
    """
    dist_values = distances[image_index].squeeze()  # Shape: (n_patches,)
    n_patches = dist_values.shape[0]
    sqrt_n_patches = int(n_patches ** 0.5)
    patch_size = H // sqrt_n_patches

    # Reshape distances to match the spatial layout of patches
    dist_map = dist_values.reshape(sqrt_n_patches, sqrt_n_patches)
    
    # Store original min/max for colorbar labels
    orig_min = dist_map.min()
    orig_max = dist_map.max()
    
    # Normalize distances for better visualization (always use full range)
    dist_normalized = (dist_map - dist_map.min()) / (dist_map.max() - dist_map.min() + 1e-8)
    
    # For cosine similarity, invert the colormap so that higher values are brighter
    if metric == 'cosine':
        # No need to invert since higher cosine similarity is better
        pass
    else:  # l2
        # Invert for L2 so that lower distances (better) are brighter
        dist_normalized = 1 - dist_normalized

    # Create patch-based heatmap (no interpolation)
    heatmap = np.zeros((H, W, 3))
    patch_colors = {}  # Store colors for each patch

    for i in range(sqrt_n_patches):
        for j in range(sqrt_n_patches):
            y0, x0 = i * patch_size, j * patch_size
            y1, x1 = y0 + patch_size, x0 + patch_size

            # Get the normalized distance for this patch
            distance_value = dist_normalized[i, j]

            # Apply the viridis colormap to this value
            color = plt.cm.viridis(distance_value)[:3]  # Get RGB values
            patch_colors[(i, j)] = color

            # Fill the patch with this color
            heatmap[y0:y1, x0:x1] = color

    metadata = {
        'min_value': orig_min, 
        'max_value': orig_max,
        'patch_colors': patch_colors,
        'metric': metric,
        'title': "L2 distance heatmap" if metric == 'l2' else "Cosine similarity heatmap"
    }
    
    return heatmap, metadata


def visualize_comprehensive_analysis(image_index, indices, distances, gen_dataset, train_dataset,
                                     n_train, gen_model, save_dir=None, metric='l2'):
    """
    Creates a comprehensive visualization with 5 subplots in the following order:
    1. Top contributor
    2. Colored sources plot
    3. Raw generated image
    4. Patch reconstruction
    5. Distance/similarity heatmap

    Args:
        image_index (int): Index of the generated image
        indices (np.ndarray): Nearest neighbor indices, shape (n_gen, n_patches, k)
        distances (np.ndarray): Distances/similarities to nearest neighbors, shape (n_gen, n_patches, k)
        gen_dataset: Dataset containing generated images
        train_dataset: Dataset containing training images
        n_train (int): Number of training images
        gen_model (str): Name of the generative model
        save_dir (str, optional): Directory to save the visualization
        metric (str): Distance metric used ('l2' or 'cosine')
    """
    # Get the generated image
    gen_img_tensor = gen_dataset[image_index][0]
    gen_img_np = gen_img_tensor.transpose(1, 2, 0)  # (H, W, C)
    H, W, C = gen_img_np.shape

    # Convert to PIL Image if it's a numpy array
    if isinstance(gen_img_np, np.ndarray):
        gen_img = Image.fromarray((gen_img_np * 255).astype(np.uint8) if gen_img_np.max() <= 1.0 else gen_img_np.astype(np.uint8))

    # 1. Extract indices for this specific image
    image_indices = indices[image_index]

    # 2. Create distance heatmap (do this first to get colors for source plot)
    dist_heatmap, heatmap_metadata = create_distance_heatmap(distances, image_index, H, W, metric)
    
    # 3. Create colored sources overlay using heatmap colors
    colored_sources, image_sources = create_colored_sources_plot(
        gen_img, 
        image_indices, 
        n_train, 
        patch_colors=heatmap_metadata['patch_colors']
    )

    # 4. Get the top contributor
    bin_count = np.bincount(image_sources, minlength=n_train)
    top_contributors = bin_count.argsort()[-10:][::-1]  # Get top 10 contributors in descending order

    most_contributing_image_idx = top_contributors[0]
    most_contributing_image = train_dataset[most_contributing_image_idx][0].transpose(1, 2, 0)  # Get training image
    if isinstance(most_contributing_image, np.ndarray):
        if most_contributing_image.max() <= 1.0:
            most_contributing_image = (most_contributing_image * 255).astype(np.uint8)
        most_contributing_image = Image.fromarray(most_contributing_image.astype(np.uint8))

    # 5. Create patch reconstruction grid
    patch_grid = create_patch_reconstruction_grid(image_indices, train_dataset, H, W)

    # Create figure with 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Plot 1: Top contributor (now first)
    axes[0].imshow(most_contributing_image)
    axes[0].set_title(f"Top Contributor ({bin_count[most_contributing_image_idx]} patches)")
    axes[0].axis('off')

    # Plot 2: Patch reconstruction
    axes[1].imshow(patch_grid)
    axes[1].set_title("Patch Reconstruction")
    axes[1].axis('off')

    # Plot 3: Raw generated image
    axes[2].imshow(gen_img)
    axes[2].set_title("Generated Image")
    axes[2].axis('off')

    # Plot 4: Colored sources
    axes[3].imshow(colored_sources)
    axes[3].set_title("Colored Sources")
    axes[3].axis('off')

    # Plot 5: Distance/similarity heatmap
    im = axes[4].imshow(dist_heatmap, cmap='viridis')
    axes[4].set_title(heatmap_metadata['title'])
    axes[4].axis('off')
    
    # Create colorbar with actual values
    cbar = plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
    if metric == 'l2':
        # For L2, lower values are better (displayed as brighter)
        cbar.ax.invert_yaxis()  # Invert colorbar
    
    # Set the colorbar labels to show the actual values
    cbar.set_ticks([0, 0.5, 1])
    if metric == 'l2':
        # For L2, show the max at the bottom, min at the top (since we inverted)
        cbar.set_ticklabels([f"{heatmap_metadata['max_value']:.2f}", 
                            f"{(heatmap_metadata['min_value'] + heatmap_metadata['max_value'])/2:.2f}", 
                            f"{heatmap_metadata['min_value']:.2f}"])
    else:
        # For cosine, show min at bottom, max at top
        cbar.set_ticklabels([f"{heatmap_metadata['min_value']:.2f}", 
                            f"{(heatmap_metadata['min_value'] + heatmap_metadata['max_value'])/2:.2f}", 
                            f"{heatmap_metadata['max_value']:.2f}"])

    plt.tight_layout()

    # Save if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/comprehensive_analysis_{gen_model}_id{image_index}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_histogram(hist, gen_model, top_k=50, save_dir=None):
    """Visualizes the patch origin histogram, i.e., the counts of how many patches came from each training image.

    Args:
        hist (array-like): Histogram data. A list, array, or similar structure representing counts
            or values to be visualized.
        gen_model (str): The name of the generative model that was used.
        top_k (int, optional): Number of top values to display in the histogram visualization.
            Defaults to 50.
        save_dir (str or None, optional): Path to the directory where the histogram plot should be
            saved. If None, the plot is not saved. Defaults to None.

    Returns:
        None
    """
    top_idxs = np.argsort(hist)[-top_k:][::-1]
    top_vals = hist[top_idxs]

    plt.figure(figsize=(10, 4))
    plt.bar(range(top_k), top_vals)
    plt.xlabel("Training Image Index")
    plt.ylabel("# Patch Matches")
    plt.tight_layout()
    # plt.ylim([0,600])
    if save_dir:
        save_path = f"{save_dir}/patch_origin_histogram_{gen_model}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def visualize_dist_histogram(distances, gen_model, save_dir=None):
    """
    Visualizes two distance histograms:
    1. The binned distances to the nearest neighbor for each patch.
    2. The binned average distances for each image (averaged over its patches).

    Args:
        distances (np.ndarray): Array of shape (n_generated, n_patches, 1)
        gen_model (str): The name of the generative model that was used.
        save_dir (str): If provided, saves plots to this directory.

    Returns:
        dict with:
            - all_patch_distances: Flattened array of all distances.
            - mean_sample_distances: Array of mean distance per generated sample.
    """
    all_patch_distances = distances.reshape(-1)
    mean_sample_distances = distances.mean(axis=1).reshape(-1)

    # Plot histogram of all patch distances
    plt.figure(figsize=(6, 4))
    plt.hist(all_patch_distances, bins=50, alpha=0.8, color='steelblue')
    plt.title(f'Patch NN Distance Histogram - {gen_model}')
    plt.xlabel('Distance')
    plt.ylabel('# Patches')
    # plt.xlim([0, 3000])
    # plt.ylim([0, 12000])
    if save_dir:
        save_path = f"{save_dir}/patch_distance_hist_{gen_model}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

    # Plot histogram of average distance per generated image
    plt.figure(figsize=(6, 4))
    plt.hist(mean_sample_distances, bins=50, alpha=0.8, color='darkorange')
    plt.title(f'Avg Patch NN Distance per Sample - {gen_model}')
    plt.xlabel('Mean Distance')
    plt.ylabel('# Generated Samples')
    # plt.xlim([500, 2500])
    plt.ylim([0, 50])
    if save_dir:
        save_path = f"{save_dir}/mean_patch_distance_per_sample_{gen_model}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

    return {
        "all_patch_distances": all_patch_distances,
        "mean_sample_distances": mean_sample_distances,
    }



@click.command()
@click.option('--top_folder',       type=str,   default="/home/shared/generative_models/recombination", help="The folder where embeddings are stored.")
@click.option('--dataset',          type=str,   default='in64', help="")
@click.option('--gen_model',        type=str,   default="edm2-img64-xl-0671088",                        help="The name of the EDM2 model.")
@click.option('--feature_model',    type=str,   default='swav-np16',                                    help="The name of the model used for feature extraction and its patch size.")
@click.option('--load_nns',         type=bool,  default=True,                                           help="Whether to load precomputed nearest neighbors.")
@click.option('--n_train',          type=int,   default=10000,                                          help="How many training images to use for the NN search.")
@click.option('--n_gen',            type=int,   default=1000,                                           help="How many generated images to use for the NN search.")
@click.option('--metric',           type=click.Choice(['l2', 'cosine']), default='cosine',
              help="Distance metric to use for nearest neighbor search.")
def main(top_folder, dataset, gen_model, feature_model, load_nns, n_train, n_gen, metric):
    embedding_dir = os.path.join(top_folder, f'embeddings/{dataset}/{feature_model}')
    save_dir = os.path.join(top_folder, f'saves/{dataset}/{metric}/{feature_model}')
    results_dir = os.path.join(top_folder, f'results/{dataset}/{metric}/{feature_model}')

    train_dir = f"{embedding_dir}/train/"
    gen_dir = f"{embedding_dir}/{gen_model}/"
    train_files = [f for f in os.listdir(train_dir) if f.startswith('features_')]
    gen_files = [f for f in os.listdir(gen_dir) if f.startswith('features_')]

    path_train = os.path.join(train_dir, sorted(train_files)[-1])  # Take file with highest number
    path_gen = os.path.join(gen_dir, sorted(gen_files)[-1])

    index_func = faiss.IndexFlatL2 if metric == 'l2' else faiss.IndexFlatIP
    create_index(path_train, n_train, save_dir, index_func=index_func)  # Creates a bigger index if needed

    print(f"\n-- {metric} {gen_model} --")
    save_path = os.path.join(save_dir, f"{gen_model}/nearest_neighbors_t{n_train}_g{n_gen}.npz")
    if load_nns and not os.path.exists(save_path):
        print(f"--load_nns was True, but no saves for this configuration exist. Recomputing them...")
        load_nns = False
    if load_nns:
        print(f"Loading nearest neighbors data from {save_path}")
        loaded = np.load(save_path)
        indices = loaded['indices']  # (n_gen, n_patches, 1)
        distances = loaded['distances']  # (n_gen, n_patches, 1)
        assert indices.shape[0] == n_gen
        assert distances.shape[0] == n_gen
    else:
        gen_embeds = load_embeddings(path_gen, n_gen).to('cuda')
        print(f"Loaded {gen_embeds.shape[0]} generated images with {gen_embeds.shape[1]} patches each.")

        indices, distances = compute_patchwise_nearest_neighbors(
            gen_embeds=gen_embeds,
            chunk_dir=os.path.join(save_dir, f"faiss_index"),
            n_train=n_train,
            index_func=index_func,
            k=1,
            use_gpu=True,
            metric=metric
        )  # (n_gen, n_patches, 1), (n_gen, n_patches, 1)

        os.makedirs(os.path.join(save_dir, gen_model), exist_ok=True)

        # Save arrays and metadata together
        np.savez(save_path,
                 indices=indices,
                 distances=distances,
                 n_train=n_train,
                 n_gen=gen_embeds.shape[0],
                 model=gen_model,
                 metric=metric,
                 path_train=path_train,
                 path_gen=path_gen)

        print(f"Saved nearest neighbors data to {save_path}")

    # Nearest
    hist = compute_patch_origin_histogram(indices, n_train=n_train)
    ent = compute_patch_origin_entropy(hist)
    uniq = compute_unique_source_count(hist)

    print(f"Patch Origin Entropy: {ent:.4f}")
    print(f"Unique Training Sources: {uniq} / {n_train}")
    os.makedirs(f"{results_dir}/{gen_model}", exist_ok=True)
    visualize_histogram(hist,
                        gen_model=gen_model,
                        save_dir=f"{results_dir}/{gen_model}")
    visualize_dist_histogram(distances,
                             gen_model=gen_model,
                             save_dir=f"{results_dir}/{gen_model}")
    print(f"Saved histograms to {results_dir}")

    # Load dataset of generated images
    gen_dataset_path = f"/home/shared/generative_models/recombination/raw_samples/{dataset}/{gen_model}"
    if "v_vae" in gen_model:
        gen_dataset_path = f"/home/shared/generative_models/recombination/raw_samples/in64/{gen_model}/50000_random_classes{gen_model.split('v_vae')[-1]}.zip"
    print(f'Loading generated dataset from {gen_dataset_path}')
    gen_dataset_kwargs = dnnlib.EasyDict(class_name='dataset.ImageFolderDataset', path=gen_dataset_path, max_size=n_gen)
    gen_dataset_obj = dnnlib.util.construct_class_by_name(**gen_dataset_kwargs)  # subclass of training.dataset.Dataset

    # Load dataset of training images
    if dataset == 'in64':
        train_dataset_path = '/home/shared/DataSets/vision_benchmarks/IN_64x64_karras/imagenet-64x64.zip'
    elif dataset == 'in512':
        train_dataset_path = '/home/shared/DataSets/vision_benchmarks/IN_512x512_karras/img512.zip'
    print(f'Loading training dataset from {train_dataset_path}')
    train_dataset_kwargs = dnnlib.EasyDict(class_name='dataset.ImageFolderDataset', path=train_dataset_path, max_size=n_train)
    train_dataset_obj = dnnlib.util.construct_class_by_name(**train_dataset_kwargs)

    for idx in tqdm(range(64), desc=f"Visual analysis"):
        visualize_comprehensive_analysis(image_index=idx,
                                         indices=indices,
                                         distances=distances,
                                         gen_dataset=gen_dataset_obj,
                                         train_dataset=train_dataset_obj,
                                         n_train=n_train,
                                         gen_model=gen_model,
                                         save_dir=f"{results_dir}/{gen_model}/comprehensive_analysis",
                                         metric=metric)
    print(f"Saved comprehensive analysis visualizations to {f'{results_dir}/{gen_model}/comprehensive_analysis/'}")

    metrics_dict = {"nearest_neighbors_indices": indices,
                    "nearest_neighbors_distances": distances,
                    "patch_origin_histogram": hist,
                    "patch_origin_entropy": ent,
                    "unique_source_count": uniq,
                    "metric": metric}
    torch.save(metrics_dict, f'{results_dir}/{gen_model}/metrics_results.pt')


### ----------- Main Execution Example -----------
if __name__ == "__main__":
    main()
