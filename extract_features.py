import torch
import os
import dnnlib
import click

from models.model_utils import FeatureExtractor
from torch_utils import distributed as dist
from tqdm import tqdm


@torch.no_grad()
def extract_features(f_extractor, data_loader, size, patch_wise=False, device='cuda'):
    if not patch_wise:
        arr = torch.zeros(size, f_extractor.f_dim, dtype=torch.float32)
    else:
        arr = torch.zeros(size, 256, f_extractor.f_dim, dtype=torch.float32)
    bs = data_loader.batch_size
    for i, (img, label) in enumerate(tqdm(data_loader)):
        img = img.to(device)
        if img.max() > 1:  # Check if images are in [0, 255] or [0,1] and adjust if needed
            img = img / 255  # [0, 255] -> [0, 1]
        assert img.min() >= 0 and img.max() <= 1, f'Input range is not [0,1]. Min: {img.min()}, Max: {img.max()}'
        feats = f_extractor.get_features(img, patch_wise=patch_wise)
        arr[bs * i:bs * (i + 1)] = feats.cpu()  # [bs, f_dim]
    return arr


def get_dataset_path(dataset):
    if dataset == 'cifar10':
        dataset_folder = '/home/shared/DataSets/cifar-10'
        dataset_path = f'{dataset_folder}/cifar10-32x32.zip'
        n_imgs = 50000
    elif dataset == 'cifar100':
        dataset_folder = '/home/shared/DataSets/cifar-100'
        dataset_path = f'{dataset_folder}/cifar100-32x32.zip'
        n_imgs = 50000
    elif dataset == 'ffhq':
        dataset_folder = '/home/shared/DataSets/vision_benchmarks/FFHQ-i'
        dataset_path = f'{dataset_folder}/ffhq-64x64.zip'
        n_imgs = 70000
    elif dataset == 'in64':
        dataset_folder = '/home/shared/DataSets/vision_benchmarks/IN_64x64_karras'
        dataset_path = f'{dataset_folder}/imagenet-64x64.zip'
        n_imgs = 1281167
    elif dataset == 'mnist':
        dataset_folder = '/home/shared/DataSets/MNIST'
        dataset_path = f'{dataset_folder}/mnist-32x32.zip'
        n_imgs = 60000
    elif dataset == 'afhqv2':
        dataset_folder = '/home/shared/DataSets/vision_benchmarks/AFHQ-v2'
        dataset_path = f'{dataset_folder}/afhqv2-64x64.zip'
        n_imgs = 15803
    elif dataset == 'cifar100-coarse':
        dataset_folder = '/home/shared/DataSets/cifar-100'
        dataset_path = f'{dataset_folder}/cifar100-coarse-32x32.zip'
        n_imgs = 50000
    elif dataset == 'cifar10_incp_804_512':
        dataset_folder = '/home/shared/DataSets/cifar-10'
        dataset_path = f'{dataset_folder}/cifar10-32x32_incp_804_512.zip'
        n_imgs = 44803
    elif dataset == 'cifar10_dino_804_512':
        dataset_folder = '/home/shared/DataSets/cifar-10'
        dataset_path = f'{dataset_folder}/cifar10-32x32_dino_804_512.zip'
        n_imgs = 39195
    elif dataset == 'cifar10_incp_dino_804_512':
        dataset_folder = '/home/shared/DataSets/cifar-10'
        dataset_path = f'{dataset_folder}/cifar10-32x32_incp_dino_804_512.zip'
        n_imgs = 48085
    elif dataset == 'in64':
        dataset_path = f'/home/shared/generative_models/recombination/raw_samples/in64/{dataset}/'
        dataset_folder = None
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
@click.option('--feature_model', type=str, default='dinov2')
@click.option('--dataset', help='Dataset Name or Folder Path', metavar='STR', type=str, default='in64', required=True)  # folder or 'mnist', 'cifar10', 'afhqv2', 'ffhq', 'imagenet'
@click.option('--batch_gpu', type=int, default=1024)
@click.option('--max_size', type=int, default=0, help='Artificially limit the size of the dataset. 0 = no limit.')
@click.option('--outdir', type=str, default="/home/shared/generative_models/recombination/embeddings/")
@click.option('--patch_wise', type=bool, default=True, help='Whether to extract features at patch level or image level.')
def main(feature_model, dataset, batch_gpu, max_size, outdir, patch_wise):
    device = 'cuda'
    f_extractor = FeatureExtractor(feature_model, device)

    # Load dataset.
    dataset_path, dataset_folder, n_imgs = get_dataset_path(dataset)
    if n_imgs is None:
        assert max_size != 0, "For custom datasets, specify the number of images using --max_size."
        n_imgs = max_size
    dist.print0(f'Loading dataset from {dataset_path}')

    dataset_kwargs = dnnlib.EasyDict(class_name='dataset.ImageFolderDataset', path=dataset_path, max_size=max_size)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=batch_gpu, **data_loader_kwargs)

    dist.print0('Extracting features...')
    features = extract_features(f_extractor,
                                dataset_loader,
                                size=min(n_imgs, max_size) if max_size != 0 else n_imgs,
                                patch_wise=patch_wise)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    save_path = os.path.join(outdir, f'{feature_model}_features.pt')
    torch.save(features, save_path)
    dist.print0(f'Saved features in ', save_path)


if __name__ == '__main__':
    main()