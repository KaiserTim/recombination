import torch
import os
import dnnlib
import click
import re
import glob

from models.model_utils import FeatureExtractor
from torch_utils import distributed as dist
from torch_utils import misc
from tqdm import tqdm


def find_latest_feature_file(outdir):
    """Find the latest feature file in the output directory.

    Args:
        outdir: Output directory to search for feature files

    Returns:
        Tuple of (file_path, num_features) or (None, 0) if no files found
    """
    if not os.path.exists(outdir):
        return None, 0

    # Look for files with pattern 'features_00000000.pt'
    files = glob.glob(os.path.join(outdir, 'features_*.pt'))
    if not files:
        return None, 0

    # Extract the number of features from each filename
    file_nums = []
    for file in files:
        match = re.search(r'features_(\d+)\.pt', os.path.basename(file))
        if match:
            file_nums.append((file, int(match.group(1))))

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
    arr = torch.zeros(full_size, n_patches, f_extractor.f_dim, dtype=torch.float32)

    # Check if we have an existing file to continue from
    starting_batch = 0
    if outdir is not None:
        latest_file, num_features = find_latest_feature_file(outdir)
        if latest_file is not None and num_features < size:
            dist.print0(f'Found existing feature file {latest_file} with {num_features} features. Continuing from there.')
            existing_features = torch.load(latest_file)
            arr[:num_features] = existing_features
            starting_batch = num_features // batch_size

    def transforms(x):
        x = torch.nn.functional.interpolate(x.to(torch.float32), size=(resize_size, resize_size), mode='bicubic', antialias=True)
        x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        return x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    for i, (img, _) in enumerate(tqdm(data_loader, desc=tqdm_desc)):
        # Skip iterations until we reach the starting batch when resuming
        if starting_batch > 0 and i < starting_batch:
            continue
            
        img = img.to(device).to(torch.float32)
        img = img / 255 if img.max() > 1 else img # Check if images are in [0, 255] or [0,1] and adjust if needed
        assert img.min() >= 0 and img.max() <= 1, f'Input range is not in [0,1]. Min: {img.min()}, Max: {img.max()}'
        img = transforms(img)  # [bs, C, 224, 244]
        bs, C, H, W = img.shape
        feats = []
        patch_size = int(H // n_patches ** 0.5)
        for j in range(0, H, patch_size):
            for k in range(0, W, patch_size):
                patch = img[:, :, j:j + patch_size, k:k + patch_size]
                out = f_extractor.get_features(patch).squeeze()  # (bs, f_dim)
                feats.append(out[:, None, :])
        feats = torch.cat(feats, dim=1)  # (bs, n_patches, f_dim)
        arr[bs * i:bs * (i + 1)] = feats.cpu()  # (bs, n_patches, f_dim)

        # Save intermediate results at specified intervals
        current_count = bs * (i + 1)
        if current_count % save_interval == 0 or current_count >= size:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            save_path = os.path.join(outdir, f'features_{current_count:08d}.pt')
            torch.save(arr[:current_count], save_path)
    dist.print0(f'Saved features in {save_path}')

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
@click.option('--feature_model', type=click.Choice(['dinov2', 'swav']), default='swav')
@click.option('--dataset', type=str, default='in64', required=True, help='Dataset Name or Folder Path')
@click.option('--batch_gpu', type=int, default=512)
@click.option('--max_size', type=int, default=0, help='Artificially limit the size of the dataset. 0 = no limit.')
@click.option('--outdir', type=str, required=True)
@click.option('--n_patches', type=int, default=16, help='Number of patches to extract from each image. Not used for DINOv2.')
@click.option('--save_interval', type=int, default=5120*2, help='Save intermediate results every N images')
def main(feature_model, dataset, batch_gpu, max_size, outdir, n_patches, save_interval):
    device = 'cuda'
    f_extractor = FeatureExtractor(feature_model, device)

    # Load dataset.
    dataset_path, dataset_folder, n_imgs = get_dataset_path(dataset)
    if n_imgs is None:
        assert max_size != 0, "For custom datasets, specify the number of images using --max_size."
        n_imgs = max_size
    n_imgs = min(n_imgs, max_size) if max_size != 0 else n_imgs
    dist.print0(f'Loading dataset from {dataset_path}')

    # Check if we have an existing file to ensure max_size is respected
    latest_file, num_features = find_latest_feature_file(outdir)
    if latest_file is not None:
        assert num_features <= n_imgs, f"Found existing feature file with {num_features} features, but max_size is {n_imgs}. Increase max_size to continue."

    dataset_kwargs = dnnlib.EasyDict(class_name='dataset.ImageFolderDataset', path=dataset_path, max_size=max_size)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=0)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=batch_gpu, **data_loader_kwargs)

    features = extract_features(f_extractor,
                                dataset_loader,
                                n_patches,
                                resize_size=448,  # Multiple of 2^n and of 14, so that crops fit before and after resizing
                                size=n_imgs,
                                tqdm_desc=f'Extracting features with {feature_model} for {n_imgs} images with {n_patches} patches per image',
                                outdir=outdir,
                                save_interval=save_interval)

    # Final save is now handled inside extract_features function
    dist.print0(f'Feature extraction complete')


if __name__ == '__main__':
    main()