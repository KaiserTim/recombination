import torch
import os
import dnnlib
import click

from models.model_utils import FeatureExtractor
from torch_utils import distributed as dist
from tqdm import tqdm


@torch.no_grad()
def extract_features(f_extractor, data_loader, n_patches, size, device='cuda'):
    arr = torch.zeros(size, n_patches, f_extractor.f_dim, dtype=torch.float32)
    for i, (img, _) in enumerate(tqdm(data_loader)):
        bs, C, H, W = img.shape
        img = img.to(device)
        img = img / 255 if img.max() > 1 else img # Check if images are in [0, 255] or [0,1] and adjust if needed
        assert img.min() >= 0 and img.max() <= 1, f'Input range is not [0,1]. Min: {img.min()}, Max: {img.max()}'
        if f_extractor.model_name == 'dinov2':
            feats = f_extractor.get_features(img)  # dinov2 returns features for each patch
        elif f_extractor.model_name == 'swav':
            feats = []
            assert n_patches ** 0.5 % 1 == 0, f"n_patches must be a perfect square. Got {n_patches}"
            patch_size = int(img.shape[-1] // n_patches ** 0.5)
            for j in range(0, H, patch_size):
                for k in range(0, W, patch_size):
                    patch = img[:, :, j:j + patch_size, k:k + patch_size]
                    out = f_extractor.get_features(patch).squeeze()  # (bs, f_dim)
                    feats.append(out[:, None, :])
            feats = torch.cat(feats, dim=1)  # (bs, n_patches, f_dim)
        arr[bs * i:bs * (i + 1)] = feats.cpu()  # (bs, n_patches, f_dim)
    return arr


def get_dataset_path(dataset):
    if dataset == 'in64':
        dataset_folder = '/home/shared/DataSets/vision_benchmarks/IN_64x64_karras'
        dataset_path = f'{dataset_folder}/imagenet-64x64.zip'
        n_imgs = 1281167
    elif dataset == 'in512':
        dataset_folder = '/home/shared/DataSets/vision_benchmarks/IN_512x512_karras'
        dataset_path = f'{dataset_folder}/imagenet-512x512.zip'  # raw images, not encoded
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
@click.option('--batch_gpu', type=int, default=1024)
@click.option('--max_size', type=int, default=0, help='Artificially limit the size of the dataset. 0 = no limit.')
@click.option('--outdir', type=str, default="/home/shared/generative_models/recombination/embeddings/")
@click.option('--n_patches', type=int, default=16, help='Number of patches to extract from each image. Not used for DINOv2.')
def main(feature_model, dataset, batch_gpu, max_size, outdir, n_patches):
    device = 'cuda'
    f_extractor = FeatureExtractor(feature_model, device)

    # Load dataset.
    dataset_path, dataset_folder, n_imgs = get_dataset_path(dataset)
    if n_imgs is None:
        assert max_size != 0, "For custom datasets, specify the number of images using --max_size."
        n_imgs = max_size
    n_imgs = min(n_imgs, max_size) if max_size != 0 else n_imgs
    dist.print0(f'Loading dataset from {dataset_path}')

    dataset_kwargs = dnnlib.EasyDict(class_name='dataset.ImageFolderDataset', path=dataset_path, max_size=max_size)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=batch_gpu, **data_loader_kwargs)

    if feature_model == 'dinov2':
        dist.print0(f"DINOv2 uses 256 patches per image. Setting n_patches to 256.")
        n_patches = 256  # DINOv2 ViT-L/14 has 224 / 14 = 16 patches per dim, 256 in total

    dist.print0(f'Extracting features with {feature_model} for {n_imgs} images with {n_patches} patches per image...')
    features = extract_features(f_extractor,
                                dataset_loader,
                                n_patches,
                                size=n_imgs)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_path = os.path.join(outdir, f'{feature_model}_features.pt')
    torch.save(features, save_path)
    dist.print0(f'Saved features in ', save_path)


if __name__ == '__main__':
    main()