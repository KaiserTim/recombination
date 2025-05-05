import os

import click
import torch
import numpy as np
import faiss
import matplotlib.pyplot as plt
import sys

from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from tqdm import tqdm



# ----------- Helper Functions -----------

def load_embeddings(path, n=-1):
    """Load .pt embeddings to the desired device (GPU by default). Limit size to n"""
    embeds = torch.load(path)
    return embeds[:n]


def flatten_embeddings(embeddings):
    """Flatten patchwise tensor from (n, 256, D) → (n*256, D)."""
    N, P, D = embeddings.shape
    return embeddings.view(-1, D), N


# ----------- A. Nearest Neighbor Patch Matches -----------

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

    Inputs:
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
    gen_flat, _ = flatten_embeddings(gen_embeds)  # Shape: (n_gen * n_patches, dim)
    gen_np = gen_flat.detach().cpu().numpy().astype('float32')

    # Prepare overall results
    all_indices = np.full((len(gen_np), k), -1, dtype=np.int32)  # Initialize to -1
    all_distances = np.full((len(gen_np), k), np.inf, dtype=np.float32)  # Initialize to infinity

    # Iterate through FAISS index chunks
    chunk_paths = sorted(glob.glob(f"{chunk_dir}/chunk_*.bin"))  # Sorted to ensure correct order
    chunk_paths = [path for path in chunk_paths if int(path.split('_')[-1].split('.')[0]) < n_train]
    print(f"Using {len(chunk_paths)} chunks from directory: {chunk_dir}")

    for chunk_idx, chunk_path in enumerate(chunk_paths):
        # Load and potentially truncate the index chunk
        index = faiss.read_index(chunk_path)
        chunk_start_idx = int(chunk_path.split('_')[-1].split('.')[0])
        if chunk_idx == len(chunk_paths) - 1:  # Trim the last chunk if needed
            entries_to_keep = (chunk_start_idx * n_patches + index.ntotal) - (n_train * n_patches)
            if entries_to_keep > 0:
                new_index = faiss.IndexFlatL2(index.d)
                new_index.add(faiss.vector_to_array(index.reconstruct_n(0, entries_to_keep)))
                index = new_index

        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        # Perform batched nearest-neighbor search on this chunk
        pbar = tqdm(range(0, len(gen_np), batch_size), desc=f"Chunk {chunk_idx + 1}/{len(chunk_paths)} NN search", dynamic_ncols=True)

        for i in pbar:
            batch = gen_np[i:i + batch_size]  # Batch of query patches
            chunk_distances, chunk_indices = index.search(batch, k)  # Perform search on this chunk

            # Adjust indices to account for chunk offset
            chunk_indices += chunk_start_idx

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


def compute_patch_origin_histogram(nn_indices, n_train):
    """
    Count how many of the 196 patches came from each training image.
    Inputs:
        nn_indices: Indices of k-nearest neighbors from the training data for each patch, (n_gen, n_patches, k)
        n_train: Number of training images
    """
    n_patches = nn_indices.shape[1]
    patch_indices = nn_indices.flatten()  # (n_gen * n_patches * k,)
    image_ids = patch_indices // n_patches  # Maps training patch indices to training image indices
    hist = np.bincount(image_ids, minlength=n_train)
    return hist


# ----------- C. Patch Origin Entropy -----------

def compute_patch_origin_entropy(hist):
    """Computes the entropy of the patch origin histogram."""
    prob = hist / np.sum(hist)
    return entropy(prob, base=2)


# ----------- D. Unique Source Count -----------

def compute_unique_source_count(hist):
    return np.count_nonzero(hist)


# ----------- E. Visualization -----------

def visualize_histogram(hist, model_name="Model", top_k=50):
    top_idxs = np.argsort(hist)[-top_k:][::-1]
    top_vals = hist[top_idxs]

    plt.figure(figsize=(10, 4))
    plt.bar(range(top_k), top_vals)
    plt.title(f"Patch Origin Histogram - {model_name}")
    plt.xlabel("Training Image Index (Top contributors)")
    plt.ylabel("# Patch Matches")
    plt.tight_layout()
    plt.savefig(f"/home/shared/generative_models/recombination/results/patch_origin_histogram_{model_name}.png",
                dpi=200, bbox_inches='tight')
    plt.show()



def create_index(path_train, n_train, save_dir, chunk_size=1024):
    """
    Save embeddings into FAISS index chunks, independent of `n_train`.
    """
    chunk_dir = os.path.join(save_dir, "faiss_index")
    os.makedirs(chunk_dir, exist_ok=True)

    train_embeds = load_embeddings(path_train, n_train)
    print(f"Loaded {train_embeds.shape[0]} training image embeddings with {train_embeds.shape[1]} patches each.")
    # pbar = tqdm(range(0, len(train_embeds), chunk_size), desc=f"Creating new FAISS index in chunks of size {chunk_size}...")
    for i in range(0, len(train_embeds), chunk_size):
        chunk_path = os.path.join(chunk_dir, f"chunk_{i:07d}.bin")
        if os.path.isfile(chunk_path):
            continue
        chunk = train_embeds[i:i + chunk_size].to('cuda')  # Slice the chunk
        chunk_index = compute_faiss_index(chunk, use_gpu=True)

        faiss.write_index(faiss.index_gpu_to_cpu(chunk_index), chunk_path)
        print(f"Saved FAISS index chunk to {chunk_path}")


@click.command()
@click.option('--embedding_folder', type=str,   default="/home/shared/generative_models/recombination/embeddings/in64", help="The folder where embeddings are stored.")
@click.option('--edm2_model',       type=str,   default="edm2-img64-xl-0671088",                                        help="The name of the EDM2 model.")
@click.option('--vae_model',        type=str,   default="v_vae_m0.0_v0.0",                                              help="The name of the VAE model.")
@click.option('--load_nns',         type=bool,  default=True,                                                           help="Whether to load precomputed nearest neighbors.")
@click.option('--save_dir',         type=str,   default="/home/shared/generative_models/recombination/saves",           help="The directory where results should be saved.")
@click.option('--n_train',          type=int,   default=10000,                                                          help="How many training images to use for the NN search.")
@click.option('--n_gen',            type=int,   default=-1,                                                             help="How many generated images to use for the NN search. '-1' = no limit")
def main(embedding_folder, edm2_model, vae_model, load_nns, save_dir, n_train, n_gen):
    path_train = f"{embedding_folder}/train/dinov2_features.pt"
    path_edm2 = f"{embedding_folder}/{edm2_model}/dinov2_features.pt"
    path_vae = f"{embedding_folder}/{vae_model}/dinov2_features.pt"

    create_index(path_train, n_train, save_dir)  # Creates a bigger index if needed

    for model_name, path_gen in [(edm2_model, path_edm2), (vae_model, path_vae)]:
        print(f"\n-- {model_name} --")
        if load_nns:
            save_path = os.path.join(save_dir, f"{model_name}/nearest_neighbors.npz")
            loaded = np.load(save_path)
            indices = loaded['indices']  # (n_gen, n_patches, 1)
            distances = loaded['distances']  # (n_gen, n_patches, 1)

            print(f"Loaded nearest neighbors data from {save_path}")
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

            os.makedirs(os.path.join(save_dir, model_name), exist_ok=True)

            # Save arrays and metadata together
            save_path = os.path.join(save_dir, f"{model_name}/nearest_neighbors.npz")
            np.savez(save_path,
                     indices=indices,
                     distances=distances,
                     n_train=n_train,
                     n_gen=gen_embeds.shape[0],
                     model=model_name,
                     path_train=path_train,
                     path_gen=path_gen)

            print(f"Saved nearest neighbors data to {save_path}")

        hist = compute_patch_origin_histogram(indices, n_train=n_train)
        ent = compute_patch_origin_entropy(hist)
        uniq = compute_unique_source_count(hist)

        print(f"Patch Origin Entropy: {ent:.4f}")
        print(f"Unique Training Sources: {uniq}")
        visualize_histogram(hist, model_name=model_name)

        metrics_dict = {}  # Initialize a dictionary to store metrics
        metrics_dict["nearest_neighbors_indices"] = indices
        metrics_dict["nearest_neighbors_distances"] = distances
        metrics_dict["patch_origin_histogram"] = hist
        metrics_dict["patch_origin_entropy"] = ent
        metrics_dict["unique_source_count"] = uniq
        torch.save(metrics_dict, '/home/shared/generative_models/recombination/results/metrics_results.pt')


### ----------- Main Execution Example -----------
if __name__ == "__main__":
    main()
