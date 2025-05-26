import torch
import os
import dnnlib
import click
import re
import glob

from model_utils import FeatureExtractor
from torch_utils import distributed as dist
from torch_utils import misc
from tqdm import tqdm


def get_all_feature_files(outdir):
    """Get all feature files in the output directory.

    Args:
        outdir: Output directory to search for feature files

    Returns:
        List of (file_path, num_features) tuples
    """
    if not os.path.exists(outdir):
        return []

    # Look for files with pattern 'features_00000000.pt'
    files = glob.glob(os.path.join(outdir, 'features_*.pt'))
    if not files:
        return []

    # Extract the number of features from each filename
    file_nums = []
    for file in files:
        match = re.search(r'features_(\d+)\.pt', os.path.basename(file))
        if match:
            file_nums.append((file, int(match.group(1))))

    return file_nums


def find_latest_feature_file(outdir):
    """Find the latest feature file in the output directory.

    Args:
        outdir: Output directory to search for feature files

    Returns:
        Tuple of (file_path, num_features) or (None, 0) if no files found
    """
    file_nums = get_all_feature_files(outdir)
    if not file_nums:
        return None, 0

    # Return the file with the highest number of features
    latest_file, num_features = max(file_nums, key=lambda x: x[1])
    return latest_file, num_features


@torch.no_grad()
def extract_features(f_extractor, data_loader, n_patches, resize_size, size, tqdm_desc, outdir, save_interval, device='cuda'):
    assert n_patches ** 0.5 % 1 == 0, f"n_patches must be a perfect square. Got {n_patches}"
    batch_size = data_loader.batch_size
    full_size = int(size // batch_size) * batch_size if size % batch_size != 0 else size

    if 'lpips' in f_extractor.model_name:
        # Need to figure out f_dim, as it depends on the input size
        img, _ = next(iter(data_loader))
        bs, C, H, W = img.shape
        patch_size = int(H // n_patches ** 0.5)
        test_in = torch.zeros(1, C, patch_size, patch_size, device=device)
        f_extractor.f_dim = f_extractor.get_features(test_in).shape[1]  # (bs, f_dim)
        dist.print0(f'LPIPS feature-dimension is {f_extractor.f_dim}')
    arr = torch.zeros(full_size, n_patches, f_extractor.f_dim, dtype=torch.float16)

    # Check if we have an existing file to continue from
    starting_batch = 0
    if outdir is not None:
        latest_file, num_features = find_latest_feature_file(outdir)
        if latest_file is not None and num_features < size:
            dist.print0(f'Found existing feature file {latest_file} with {num_features} features. Continuing from there.')
            existing_features = torch.load(latest_file)
            arr[:num_features] = existing_features
            starting_batch = num_features // batch_size

    def transforms(x, feature_model):
        """Apply ImageNet normalization"""
        x = torch.nn.functional.interpolate(x.to(torch.float32), size=(resize_size, resize_size), mode='bicubic', antialias=True)

        if 'lpips' in feature_model:
            x = x - misc.const_like(x, [0.5, 0.5, 0.5]).reshape(1, -1, 1, 1)
            x = x / misc.const_like(x, [0.5, 0.5, 0.5]).reshape(1, -1, 1, 1)
        else:
            x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
            x = x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
        return x

    last_saved_file = None
    
    for i, (img, _) in enumerate(tqdm(data_loader, desc=tqdm_desc)):
        # Skip iterations until we reach the starting batch when resuming
        if starting_batch > 0 and i < starting_batch:
            continue
            
        img = img.to(device).to(torch.float32)
        img = img / 255 if img.max() > 1 else img # Check if images are in [0, 255] or [0,1] and adjust if needed
        assert img.min() >= 0 and img.max() <= 1, f'Input range is not in [0,1]. Min: {img.min()}, Max: {img.max()}'
        img = transforms(img, f_extractor.model_name)
        feats = []

        # For each patch in the image
        for j in range(0, H, patch_size):
            for k in range(0, W, patch_size):
                patch = img[:, :, j:j + patch_size, k:k + patch_size]
                
                # Extract features for the current patch
                out = f_extractor.get_features(patch).to(torch.float16)  # (bs, f_dim)
                
                # Handle different output shapes depending on model
                if len(out.shape) > 2:
                    out = out.squeeze()
                
                feats.append(out[:, None, :])
                
        feats = torch.cat(feats, dim=1)  # (bs, n_patches, f_dim)
        arr[bs * i:bs * (i + 1)] = feats.cpu()  # (bs, n_patches, f_dim)

        # Save intermediate results at specified intervals
        current_count = bs * (i + 1)
        if current_count % save_interval == 0 or current_count >= size:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            save_path = os.path.join(outdir, f'features_{current_count:08d}.pt')

            # Save the new file
            try:
                torch.save(arr[:current_count], save_path)
                save_successful = True
            except Exception as e:
                dist.print0(f'Error saving file {save_path}: {e}')
                save_successful = False

            # Clean up previous feature files only after a successful save
            if save_successful and save_path != last_saved_file:  # Ensure we're not deleting the file we just saved
                all_feature_files = get_all_feature_files(outdir)
                for file_path, _ in all_feature_files:
                    # Keep the file we just saved, delete all others
                    if file_path != save_path:
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            dist.print0(f'Warning: Failed to delete previous feature file {file_path}: {e}')
                
                last_saved_file = save_path
    
    if last_saved_file:
        dist.print0(f'Saved features in {last_saved_file}')

    return arr


def get_dataset_path(dataset):
    if dataset == 'in64':
        dataset_folder = '/home/shared/DataSets/vision_benchmarks/IN_64x64_karras'
        dataset_path = f'{dataset_folder}/imagenet-64x64.zip'
        n_imgs = 1281167
    elif dataset == 'in512':
        dataset_folder = '/home/shared/DataSets/vision_benchmarks/IN_512x512_karras'
        dataset_path = f'{dataset_folder}/img512.zip'  # raw images, not encoded
        n_imgs = 1281167
    elif os.path.isdir(dataset):
        dataset_path = dataset_folder = dataset
        n_imgs = 0  # Count the images in this folder
        for root, dirs, files in os.walk(dataset_path):
            images = [file for file in files if file.endswith('.png')]
            n_imgs += len(images)
    else:
        dataset_path = dataset_folder = dataset
        n_imgs = None  # Needs to be manually set for .zip files
    return dataset_path, dataset_folder, n_imgs


@click.command()
@click.option('--feature_model', type=click.Choice(['dinov2', 'swav', 'lpips-alex', 'lpips-vgg', 'lpips-squeeze', 'dists']), required=True)
@click.option('--dataset', type=str, default='in64', required=True, help='Dataset Name or Folder Path')
@click.option('--batch_gpu', type=int, default=512)
@click.option('--max_size', type=int, default=0, help='Artificially limit the size of the dataset. 0 = no limit.')
@click.option('--outdir', type=str, required=True)
@click.option('--n_patches', type=int, default=16, help='Number of patches to extract from each image.')
@click.option('--save_interval', type=int, default=5120*2, help='Save intermediate results every N images')
def main(feature_model, dataset, batch_gpu, max_size, outdir, n_patches, save_interval):
    device = 'cuda'
    
    # Initialize feature extractor with appropriate model
    f_extractor = FeatureExtractor(feature_model, device)

    # Load dataset
    dataset_path, dataset_folder, n_imgs = get_dataset_path(dataset)
    if n_imgs is None:
        assert max_size != 0, "For custom datasets, specify the number of images using --max_size."
        n_imgs = max_size
    n_imgs = min(n_imgs, max_size) if max_size != 0 else n_imgs
    dist.print0(f'Loading dataset from {dataset_path}')

    # Check if we have an existing file to ensure max_size is respected
    latest_file, num_features = find_latest_feature_file(outdir)
    if latest_file is not None:
        print(outdir, n_imgs, num_features, max_size)
        assert num_features <= n_imgs, f"Found existing feature file with {num_features} features, but max_size is {n_imgs}. Increase max_size to continue."

    dataset_kwargs = dnnlib.EasyDict(class_name='dataset.ImageFolderDataset', path=dataset_path, max_size=max_size)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=0)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=batch_gpu, **data_loader_kwargs)

    # Set appropriate resize resolution based on feature model
    if feature_model == 'dinov2':
        resize_size = 448  # divisible by 14 and powers of 2, so that e.g. 16 or 64 patches result in patch sizes divisible by 14
    elif feature_model == 'swav':
        resize_size = 224  # native resolution for swav, matches the inductive bias of the model w.r.t. object scale
    elif 'lpips' in feature_model or feature_model == 'dists':
        resize_size = 512  # Higher resolution for perceptual metrics

    # Extract features
    dist.print0(f'Extracting features with {feature_model} for {n_imgs} images with {n_patches} patches per image')
    features = extract_features(f_extractor,
                                dataset_loader,
                                n_patches,
                                resize_size=resize_size,
                                size=n_imgs,
                                tqdm_desc=f'Extracting {feature_model} features',
                                outdir=outdir,
                                save_interval=save_interval)

    # Final save is handled inside extract_features function
    dist.print0(f'Feature extraction complete')


if __name__ == '__main__':
    main()