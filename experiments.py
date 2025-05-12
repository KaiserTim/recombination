import os
import click
import torch
import numpy as np
import faiss
import matplotlib.pyplot as plt
import sys
import torchvision.transforms.functional as TF
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


def flatten_embeddings(embeddings):
    """Flatten patchwise tensor from (n, 256, D) → (n*256, D).
    
    Args:
        embeddings (torch.Tensor): Input tensor of shape (n, 256, D).
    
    Returns:
        tuple: Contains:
            - torch.Tensor: Flattened tensor of shape (n*256, D)
            - int: Original number of images n
    """
    N, P, D = embeddings.shape
    return embeddings.view(-1, D), N


# ----------- A. Nearest Neighbor Patch Matches -----------

def create_index(path_train, n_train, save_dir, chunk_size=1024):
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
    assert os.path.isdir(chunk_dir), f"Chunk dir doesn't exist: {chunk_dir}"
    os.makedirs(chunk_dir, exist_ok=True)

    train_embeds = None
    # pbar = tqdm(range(0, n_train, chunk_size), desc=f"Creating new FAISS index in chunks of size {chunk_size}...")
    for i in range(0, n_train, chunk_size):
        chunk_path = os.path.join(chunk_dir, f"chunk_{i:07d}.bin")
        if os.path.isfile(chunk_path):
            continue
        if train_embeds is None:
            train_embeds = load_embeddings(path_train, n_train)
            print(f"Loaded {train_embeds.shape[0]} training image embeddings with {train_embeds.shape[1]} patches each.")
        chunk = train_embeds[i:i + chunk_size].to('cuda')  # Slice the chunk
        chunk_index = compute_faiss_index(chunk, use_gpu=True)

        faiss.write_index(faiss.index_gpu_to_cpu(chunk_index), chunk_path)
        print(f"Saved FAISS index chunk to {chunk_path}")


def compute_faiss_index(data, use_gpu, batch_size=1024):
    """
    Create a FAISS index for nearest neighbor search.
    Inputs:
        data: Embeddings of image patches, (n_imgs, n_patches, D)
        use_gpu: Whether to use GPU for FAISS index
        batch_size: Batch size for FAISS search (higher values will increase search speed but require more memory)
    Returns:
        all_indices: Indices of k-nearest neighbors from the training data for each patch, (n_gen, n_patches, k)
        all_distances: The corresponding distances, (n_gen, n_patches, k)
    """
    n_gen, n_patches, _ = data.shape
    train_flat, n_imgs = flatten_embeddings(data)  # (n_imgs * n_patches, D)

    # Convert to float32 numpy arrays
    train_np = train_flat.detach().cpu().numpy().astype('float32')

    embed_dim = train_np.shape[1]

    def add_to_faiss_index(index, data, batch_size, use_gpu):
        """Helper function to add batches of data to the FAISS index with a progress bar."""
        pbar = tqdm(total=len(data), desc=f"Building FAISS index with {n_gen} images and {n_patches} patches each.", file=sys.stdout, dynamic_ncols=True, leave=True)
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            index.add(batch)
            if use_gpu:  # Display GPU memory usage if applicable
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # In GB
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # In GB
                pbar.set_postfix({"GPU Alloc (GB)": f"{allocated:.2f}", "GPU Resrv (GB)": f"{reserved:.2f}"})
            pbar.update(len(batch))

        pbar.close()

    # Build FAISS index
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(embed_dim)
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        index = faiss.IndexFlatL2(embed_dim)

    add_to_faiss_index(index, train_np, batch_size, use_gpu)
    assert index.ntotal == n_imgs * n_patches, "Index size does not match the number of training patches."
    return index


def compute_patchwise_nearest_neighbors(
        gen_embeds,  # Generated embeddings, shape (n_gen, n_patches, D)
        chunk_dir,  # Directory containing the FAISS index chunks
        n_train,  # Number of total training images to use
        k=1,  # Number of nearest neighbors to return
        use_gpu=True,  # Use GPU if available
        batch_size=16,  # Batch size for FAISS search
):
    """
    Compute the nearest neighbors of each generated patch using FAISS index chunks.
    Instead of loading the entire FAISS index into memory, process one index chunk at a time.

    Args:
        gen_embeds: Embeddings of generated patches, shape (n_gen, n_patches, D).
        chunk_dir: Path to the directory containing FAISS index chunks.
        n_train_patches_per_chunk: Number of training patches contained in each FAISS index chunk.
        k: The number of nearest neighbors to return.
        use_gpu: Whether to use GPU for FAISS index processing.
        batch_size: Batch size for FAISS search.

    Returns:
        all_indices: Indices of k-nearest neighbors for each generated patch, shape (n_gen, n_patches, k).
        all_distances: Corresponding distances of k-nearest neighbors, shape (n_gen, n_patches, k).
    """
    import glob

    # Flatten generated embeddings
    n_gen, n_patches, dim = gen_embeds.shape
    print(f"Performing NN search for {n_gen} images with {n_patches} patches each...")
    gen_flat, _ = flatten_embeddings(gen_embeds)  # Shape: (n_gen * n_patches, embed_dim)
    gen_np = gen_flat.detach().cpu().numpy().astype('float32')

    # Prepare overall results
    all_indices = np.full((len(gen_np), k), -1, dtype=np.int32)  # Initialize to -1
    all_distances = np.full((len(gen_np), k), np.inf, dtype=np.float32)  # Initialize to infinity

    # Iterate through FAISS index chunks
    chunk_paths = sorted(glob.glob(f"{chunk_dir}/chunk_*.bin"))  # Sorted to ensure correct order
    chunk_paths = [path for path in chunk_paths if int(path.split('_')[-1].split('.')[0]) < n_train]
    print(f"Using {len(chunk_paths)} chunks from {n_train} training images from directory: {chunk_dir}")

    pbar = tqdm(enumerate(chunk_paths), total=len(chunk_paths), desc="", dynamic_ncols=True)
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
                new_index = faiss.IndexFlatL2(index.d)
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

            # Update overall results (keep closest neighbors across chunks)
            for j in range(len(batch)):
                # Combine neighbors across chunks with distance-based sorting
                combined_distances = np.concatenate((all_distances[i + j], chunk_distances[j]))
                combined_indices = np.concatenate((all_indices[i + j], chunk_indices[j]))
                sorted_indices = np.argsort(combined_distances)[:k]  # Keep top-k
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
    plt.ylim([0,600])
    if save_dir:
        save_path = f"{save_dir}/patch_origin_histogram_{gen_model}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved patch origin histogram to {save_path}")
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
    plt.xlim([0, 3000])
    plt.ylim([0, 12000])
    if save_dir:
        save_path = f"{save_dir}/patch_distance_hist_{gen_model}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved distance histogram to {save_path}")
    plt.show()

    # Plot histogram of average distance per generated image
    plt.figure(figsize=(6, 4))
    plt.hist(mean_sample_distances, bins=50, alpha=0.8, color='darkorange')
    plt.title(f'Avg Patch NN Distance per Sample - {gen_model}')
    plt.xlabel('Mean Distance')
    plt.ylabel('# Generated Samples')
    plt.xlim([500, 2500])
    plt.ylim([0, 50])
    if save_dir:
        save_path = f"{save_dir}/mean_patch_distance_per_sample_{gen_model}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved average distance per sample histogram to {save_path}")
    plt.show()

    return {
        "all_patch_distances": all_patch_distances,
        "mean_sample_distances": mean_sample_distances,
    }


def visualize_patch_sources(image_index, indices, gen_dataset, train_dataset, n_train, gen_model, alpha=0.15, save_dir=None):
    """
    Overlay a color-coded patch source map over a generated image.

    Args:
        image_index (int): Index of the generated image to visualize.
        indices (np.ndarray): Nearest neighbor indices of shape (n_gen, n_patches, 1).
        gen_dataset: A dataset object with the generated images returning PIL image or tensor.
        train_dataset: A dataset object with the training images returning PIL image or tensor.
        n_train (int): Total number of training images.
        alpha (float): Opacity for the overlay colors.
    """
    image = gen_dataset[image_index][0].transpose(1,2,0)  # discard the label, (H, W, C)
    H, W, C = image.shape
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Get patch-level source IDs
    patch_sources = indices[image_index].squeeze()  # Shape: (n_patches,)
    n_patches = patch_sources.shape[0]
    patch_size = int(H / n_patches ** 0.5)  # Figure out patch dimensions within the generated image
    image_sources = patch_sources // n_patches  # Map patch IDs to image IDs

    # Summarize top contributors
    bin_count = np.bincount(image_sources, minlength=n_train)
    top_contributors = bin_count.argsort()[-10:][::-1]  # Get top 10 contributors in descending order
    top_contributors_summary = ", ".join(f"{idx}: {bin_count[idx]}" for idx in top_contributors if bin_count[idx] > 0)
    print(f"Top patch contributors: {top_contributors_summary}")

    # Normalize training IDs to [0, 1] for colormap
    normed_sources = image_sources / n_train
    colors = cm['hsv'](normed_sources)[:, :3]  # RGB colors

    # Create overlay
    image = image.convert("RGBA")  # Ensure image is in RGBA format for blending
    overlay = Image.new("RGBA", image.size)  # Create a blank overlay
    draw = ImageDraw.Draw(overlay, "RGBA")

    sqrt_n_patches = int(n_patches ** 0.5)
    for i in range(sqrt_n_patches):
        for j in range(sqrt_n_patches):
            patch_idx = i * sqrt_n_patches + j
            color = tuple(int(c * 255) for c in colors[patch_idx]) + (int(255 * alpha),)
            x0, y0 = j * patch_size, i * patch_size
            x1, y1 = x0 + patch_size, y0 + patch_size
            draw.rectangle([x0, y0, x1, y1], fill=color)

    # Blend the overlay with the original image using alpha compositing
    image_with_overlay = Image.alpha_composite(image, overlay)

    # Get the training image with the most contributions
    most_contributing_image_idx = top_contributors[0]
    most_contributing_image = train_dataset[most_contributing_image_idx][0].transpose(1, 2, 0)  # Get training image
    if isinstance(most_contributing_image, np.ndarray):
        most_contributing_image = Image.fromarray(most_contributing_image)

    # Plot side by side: Patch overlay and contributing training image
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Plot the generated image with the patch overlay
    for ax, img, title in zip(axes, [image_with_overlay, most_contributing_image], ["Generated Image", "Top Contributor"]):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    if save_dir:
        save_path = f"{save_dir}/colored_sources_{gen_model}_id{image_index}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.show()



def visualize_patch_match_grid(image_index, gen_dataset, train_dataset, indices, gen_model, save_dir=None):
    """
    Display a generated image and a 14x14 grid of its nearest neighbor patches.

    Args:
        image_index (int): Index of the generated image.
        gen_dataset: A dataset of generated images.
        train_dataset: A dataset of training images.
        indices (np.ndarray): Nearest neighbor indices, shape (n_gen, n_patches, 1).
        gen_model: The name of the generative model.
        save_dir: If provided, saves plots to this directory.
    """
    # Get the generated image
    gen_img = gen_dataset[image_index][0].transpose(1, 2, 0)  # (H, W, C)
    H, W, C = gen_img.shape
    if isinstance(gen_img, torch.Tensor):
        gen_img = TF.to_pil_image(gen_img)

    # Extract nearest neighbor patch indices
    patch_sources = indices[image_index].squeeze()  # Shape: (n_patches,)
    n_patches = patch_sources.shape[0]
    sqrt_n_patches = int(n_patches ** 0.5)
    patch_size = H // sqrt_n_patches  # Calculate patch size based on generated image size
    image_sources = patch_sources // n_patches  # Get corresponding training image IDs

    # Extract patches from training images
    matched_patches = []
    for i, train_img_idx in enumerate(image_sources[:n_patches]):  # Limit to grid size
        train_image = train_dataset[train_img_idx][0].transpose(1, 2, 0)  # Load training image and convert
        y = (i // sqrt_n_patches) * patch_size
        x = (i % sqrt_n_patches) * patch_size
        patch = train_image[y:y + patch_size, x:x + patch_size, :]
        matched_patches.append(torch.from_numpy(patch).permute(2, 0, 1))  # Convert back to tensor format

    # Create a grid of matched patches
    patch_grid = make_grid(matched_patches, nrow=sqrt_n_patches, padding=0).numpy()  # (C, H, W)

    # Plot side-by-side
    plt.figure(figsize=(10, 5))

    # Plot generated image
    plt.subplot(1, 2, 1)
    plt.imshow(gen_img)
    plt.title(f"Generated Image {image_index}")
    plt.axis('off')

    # Plot grid of nearest neighbor patches
    plt.subplot(1, 2, 2)
    plt.imshow(patch_grid.transpose(1, 2, 0))
    plt.title("Nearest Patch Matches")
    plt.axis('off')

    # Save and show the plot
    plt.tight_layout()
    if save_dir:
        save_path = f"{save_dir}/patch_reconstruction_{gen_model}_id{image_index}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved patch reconstruction image to {save_path}")
    plt.show()


@click.command()
@click.option('--embedding_folder', type=str,   default="/home/shared/generative_models/recombination/embeddings/in64", help="The folder where embeddings are stored.")
@click.option('--gen_model',        type=str,   default="edm2-img64-xl-0671088",                                        help="The name of the EDM2 model.")
@click.option('--load_nns',         type=bool,  default=True,                                                           help="Whether to load precomputed nearest neighbors.")
@click.option('--save_dir',         type=str,   default="/home/shared/generative_models/recombination/saves",           help="The directory where results should be saved.")
@click.option('--n_train',          type=int,   default=10000,                                                          help="How many training images to use for the NN search.")
@click.option('--n_gen',            type=int,   default=-1,                                                             help="How many generated images to use for the NN search. '-1' = no limit")
def main(embedding_folder, gen_model, load_nns, save_dir, n_train, n_gen):
    path_train = f"{embedding_folder}/train/dinov2_features.pt"
    path_gen = f"{embedding_folder}/{gen_model}/dinov2_features.pt"

    create_index(path_train, n_train, save_dir)  # Creates a bigger index if needed

    print(f"\n-- {gen_model} --")
    if load_nns:
        save_path = os.path.join(save_dir, f"{gen_model}/nearest_neighbors_t{n_train}_g{n_gen}.npz")
        print(f"Loading nearest neighbors data from {save_path}")
        loaded = np.load(save_path)
        indices = loaded['indices']  # (n_gen, n_patches, 1)
        distances = loaded['distances']  # (n_gen, n_patches, 1)
        assert indices.shape == (n_gen, 256, 1), f"Wrong indices shape: {indices.shape}"
        assert distances.shape == (n_gen, 256, 1), f"Wrong distances shape: {distances.shape}"
    else:
        gen_embeds = load_embeddings(path_gen, n_gen).to('cuda')
        print(f"Loaded {gen_embeds.shape[0]} generated images with {gen_embeds.shape[1]} patches each.")

        indices, distances = compute_patchwise_nearest_neighbors(
            gen_embeds=gen_embeds,
            chunk_dir=os.path.join(save_dir, f"faiss_index"),
            n_train=n_train,
            k=1,
            use_gpu=True
        )  # (n_gen, n_patches, 1), (n_gen, n_patches, 1)

        os.makedirs(os.path.join(save_dir, gen_model), exist_ok=True)

        # Save arrays and metadata together
        save_path = os.path.join(save_dir, f"{gen_model}/nearest_neighbors_t{n_train}_g{n_gen}.npz")
        np.savez(save_path,
                 indices=indices,
                 distances=distances,
                 n_train=n_train,
                 n_gen=gen_embeds.shape[0],
                 model=gen_model,
                 path_train=path_train,
                 path_gen=path_gen)

        print(f"Saved nearest neighbors data to {save_path}")

    # Nearest
    hist = compute_patch_origin_histogram(indices, n_train=n_train)
    ent = compute_patch_origin_entropy(hist)
    uniq = compute_unique_source_count(hist)

    print(f"Patch Origin Entropy: {ent:.4f}")
    print(f"Unique Training Sources: {uniq} / {n_train}")
    os.makedirs(f"/home/shared/generative_models/recombination/results/{gen_model}", exist_ok=True)
    visualize_histogram(hist,
                        gen_model=gen_model,
                        save_dir=f"/home/shared/generative_models/recombination/results/{gen_model}")
    visualize_dist_histogram(distances,
                             gen_model=gen_model,
                             save_dir=f"/home/shared/generative_models/recombination/results/{gen_model}")

    # Load dataset of generated images
    gen_dataset_path = f"/home/shared/generative_models/recombination/raw_samples/in64/{gen_model}"
    if "v_vae" in gen_model:
        gen_dataset_path = f"/home/shared/generative_models/recombination/raw_samples/in64/{gen_model}/50000_random_classes{gen_model.split('v_vae')[-1]}.zip"
    print(f'Loading generated dataset from {gen_dataset_path}')
    gen_dataset_kwargs = dnnlib.EasyDict(class_name='dataset.ImageFolderDataset', path=gen_dataset_path, max_size=n_gen)
    gen_dataset_obj = dnnlib.util.construct_class_by_name(**gen_dataset_kwargs)  # subclass of training.dataset.Dataset

    # Load dataset of training images
    train_dataset_path = '/home/shared/DataSets/vision_benchmarks/IN_64x64_karras/imagenet-64x64.zip'
    print(f'Loading training dataset from {train_dataset_path}')
    train_dataset_kwargs = dnnlib.EasyDict(class_name='dataset.ImageFolderDataset', path=train_dataset_path, max_size=n_train)
    train_dataset_obj = dnnlib.util.construct_class_by_name(**train_dataset_kwargs)

    for idx in range(5):
        visualize_patch_sources(image_index=idx,
                                indices=indices,
                                gen_dataset=gen_dataset_obj,
                                train_dataset=train_dataset_obj,
                                n_train=n_train,
                                gen_model=gen_model,
                                save_dir=f"/home/shared/generative_models/recombination/results/{gen_model}")

        visualize_patch_match_grid(image_index=idx,
                                   gen_dataset=gen_dataset_obj,
                                   train_dataset=train_dataset_obj,
                                   indices=indices,
                                   gen_model=gen_model,
                                   save_dir=f"/home/shared/generative_models/recombination/results/{gen_model}")

    metrics_dict = {"nearest_neighbors_indices": indices,
                    "nearest_neighbors_distances": distances,
                    "patch_origin_histogram": hist,
                    "patch_origin_entropy": ent,
                    "unique_source_count": uniq}
    torch.save(metrics_dict, f'/home/shared/generative_models/recombination/results/{gen_model}/metrics_results.pt')


### ----------- Main Execution Example -----------
if __name__ == "__main__":
    main()
