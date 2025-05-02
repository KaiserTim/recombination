import os
import torch
import numpy as np
import faiss
import matplotlib.pyplot as plt
import sys

from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from tqdm import tqdm



# ----------- Helper Functions -----------

def load_embeddings(path, device="cuda"):
    """Load .pt embeddings to the desired device (GPU by default)."""
    return torch.load(path, map_location=device)


def flatten_embeddings(embeddings):
    """Flatten patchwise tensor from (n, 256, D) → (n*256, D)."""
    N, P, D = embeddings.shape
    return embeddings.view(-1, D), N


def save_plot(figure, save_path):
    figure.savefig(save_path, dpi=200, bbox_inches='tight')


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
    print(f"Building FAISS index with {n_gen} images and {n_patches} patches each.")
    train_flat, n_imgs = flatten_embeddings(data)  # (n_imgs * n_patches, D)

    # Convert to float32 numpy arrays
    train_np = train_flat.detach().cpu().numpy().astype('float32')

    embed_dim = train_np.shape[1]

    def add_to_faiss_index(index, data, batch_size, use_gpu):
        """Helper function to add batches of data to the FAISS index with a progress bar."""
        pbar = tqdm(total=len(data), desc="Building FAISS index", file=sys.stdout, dynamic_ncols=True, leave=True)
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


def compute_patchwise_nearest_neighbors(generated, index, k=1, use_gpu=True, batch_size=16):
    """
    Compute the nearest neighbor of each generated patch from the training patches.
    Inputs:
        generated: Embeddings of generated patches, (n_gen, n_patches, D)
        index: Pre-computed FAISS index of training patches, (n_imgs, n_patches, D)
        k: Number of nearest neighbors to return
        use_gpu: Whether to use GPU for FAISS index
        batch_size: Batch size for FAISS search (higher values will increase search speed but require more memory)
    Returns:
        all_indices: Indices of k-nearest neighbors from the training data for each patch, (n_gen, n_patches, k)
        all_distances: The corresponding distances, (n_gen, n_patches, k)
    """
    n_gen, n_patches, _ = generated.shape
    print(f"Determining {k}-nearest neighbors for {n_gen} generated images with {n_patches} patches each.")
    gen_flat, _ = flatten_embeddings(generated)  # (n_gen * n_patches, D)
    gen_np = gen_flat.detach().cpu().numpy().astype('float32')

    all_indices = []
    all_distances = []

    pbar = tqdm(range(0, len(gen_np), batch_size), desc="FAISS NN search", dynamic_ncols=True, leave=True)
    for i in pbar:
        batch = gen_np[i:i + batch_size]
        distances, indices = index.search(batch, k)
        all_indices.append(indices)
        all_distances.append(distances)

        if use_gpu:  # Display GPU memory usage if applicable
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # In GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # In GB
            pbar.set_postfix({"GPU Alloc (GB)": f"{allocated:.2f}", "GPU Resrv (GB)": f"{reserved:.2f}"})

    all_indices = np.concatenate(all_indices, axis=0).astype('int32').reshape(n_gen, n_patches, k)
    all_distances = np.concatenate(all_distances, axis=0).astype('int32').reshape(n_gen, n_patches, k)

    return all_indices, all_distances


# ----------- B. Patch Origin Histogram -----------

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
    save_plot(f"/home/shared/generative_models/recombination/results/patch_origin_histogram_{model_name}.png")
    plt.show()


### ----------- Main Execution Example -----------
# todo make this a click script
if __name__ == "__main__":
    embedding_folder = "/home/shared/generative_models/recombination/embeddings/in64"
    edm2_model = "edm2-img64-xs-2147483"
    vae_model = "v_vae_m0.0_v0.0"
    path_train = f"{embedding_folder}/train/dinov2_features.pt"
    path_edm2 = f"{embedding_folder}/{edm2_model}/dinov2_features.pt"
    path_vae = f"{embedding_folder}/{vae_model}/dinov2_features.pt"

    train_embeds = load_embeddings(path_train)
    print(f"Loaded {train_embeds.shape[0]} training images with {train_embeds.shape[1]} patches each.")
    edm2_embeds = load_embeddings(path_edm2)
    print(f"Loaded {edm2_embeds.shape[0]} edm2 images with {edm2_embeds.shape[1]} patches each.")
    vae_embeds = load_embeddings(path_vae)
    print(f"Loaded {vae_embeds.shape[0]} vae images with {vae_embeds.shape[1]} patches each.")

    train_index = compute_faiss_index(train_embeds, use_gpu=True)

    load_nns = False
    save_dir = "/home/shared/generative_models/recombination/saves"

    for model_name, gen_embeds in [(edm2_model, edm2_embeds), (vae_model, vae_embeds)]:
        print(f"\n-- {model_name} --")
        if load_nns:
            indices_path = os.path.join(save_dir, f"{model_name}/indices.npy")
            distances_path = os.path.join(save_dir, f"{model_name}/distances.npy")

            indices = np.load(indices_path)
            distances = np.load(distances_path)

            print(f"Loaded indices from {indices_path} and distances from {distances_path}")
        else:
            indices, distances = compute_patchwise_nearest_neighbors(
                gen_embeds,
                train_index,
                k=1,
                use_gpu=True
            )  # (n_gen, n_patches, 1), (n_gen, n_patches, 1)

            os.makedirs(os.path.join(save_dir, model_name), exist_ok=True)
            indices_path = os.path.join(save_dir, f"{model_name}/indices.npy")
            distances_path = os.path.join(save_dir, f"{model_name}/distances.npy")

            # Save to .npy files
            np.save(indices_path, indices)
            np.save(distances_path, distances)

            print(f"Saved indices to {indices_path} and distances to {distances_path}")

        hist = compute_patch_origin_histogram(indices, n_train=train_embeds.shape[0])
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
