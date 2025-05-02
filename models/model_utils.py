import torch
import pickle
import dnnlib
import timm
# import open_clip

from torch_utils import distributed as dist, misc
from models import vision_transformer as vits
from models import utils_mk
from torchvision.transforms import InterpolationMode, Resize, CenterCrop, Normalize, Compose


def load_inception(device='cuda'):
    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    with dnnlib.util.open_url(detector_url, verbose=True) as f:
        f_model = pickle.load(f).to(device)
    f_model.eval()

    def transforms(tensor):
        return tensor * 255

    f_dim = 2048

    return f_model, transforms, f_dim

def load_convnext_base(device='cuda'):
    # https://github.com/facebookresearch/ConvNeXt
    import models.convnext as convnext  # Causes an error when using clip, if imported at the top
    f_model = convnext.__dict__['convnext_base'](pretrained=True, in_22k=True, num_classes=21841).to(device)
    f_model.eval()
    transforms = Compose([Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
                          CenterCrop(size=(224, 224)),
                          Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                    std=torch.tensor([0.229, 0.224, 0.225]))])

    f_dim = 1024

    return f_model, transforms, f_dim

def load_dino_vit_base(device='cuda'):
    f_model = vits.__dict__['vit_base'](img_size=[224], patch_size=16, num_classes=0)
    utils_mk.load_pretrained_weights(f_model, "", "teacher", 'vit_base', 16)
    f_model = f_model.to(device)
    f_model.eval()
    transforms = Compose([Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
                          CenterCrop(size=(224, 224)),
                          Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                    std=torch.tensor([0.229, 0.224, 0.225]))])

    f_dim = 768

    return f_model, transforms, f_dim

def load_sup_vit_base(device='cuda'):
    f_model = timm.create_model("vit_base_patch16_224.augreg_in1k", in_chans=3, num_classes=-1, pretrained=True)
    f_model = f_model.to(device)
    f_model.eval()
    transforms = Compose([Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
                          CenterCrop(size=(224, 224)),
                          Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                    std=torch.tensor([0.229, 0.224, 0.225]))])

    f_dim = 768

    return f_model, transforms, f_dim

def load_clip_convnext_base(device='cuda'):
    # https://github.com/mlfoundations/open_clip
    f_model, _, _ = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
    f_model = f_model.to(device)
    f_model.eval()
    transforms = Compose([Resize(size=256, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
                          CenterCrop(size=(256, 256)),
                          Normalize(mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
                                    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]))])

    f_dim = 640

    return f_model, transforms, f_dim

def load_clip_vit_base(device='cuda'):
    # https://github.com/mlfoundations/open_clip
    f_model, _, t = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    f_model = f_model.to(device)
    f_model.eval()
    transforms = Compose([Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
                          CenterCrop(size=(224, 224)),
                          Normalize(mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
                                    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]))])

    f_dim = 512

    return f_model, transforms, f_dim

def load_dinov2(device='cuda'):
    import warnings
    warnings.filterwarnings('ignore', 'xFormers is not available')
    torch.hub.set_dir(dnnlib.make_cache_dir_path('torch_hub'))
    f_model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitl14', trust_repo=True, verbose=False, skip_validation=True)
    f_model = f_model.to(device)
    f_model.eval().requires_grad_(False)

    def transforms(x):
        x = torch.nn.functional.interpolate(x.to(torch.float32), size=(224, 224), mode='bicubic', antialias=True)

        # Adjust dynamic range.
        x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        return x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    f_dim = 1024

    return f_model, transforms, f_dim


class FeatureExtractor:
    def __init__(self, model_name, device='cuda'):
        self.model_name = model_name
        self.device = device
        self.f_model, self.transforms, self.f_dim = self.load_feature_model()

    def load_feature_model(self):
        if self.model_name == 'inception':
            return load_inception(self.device)
        elif self.model_name == 'vit_base_dino':
            return load_dino_vit_base(self.device)
        elif self.model_name == 'vit_base_clip':
            return load_clip_vit_base(self.device)
        elif self.model_name == 'vit_base_sup':
            return load_sup_vit_base(self.device)
        elif self.model_name == 'convnext_base_clip':
            return load_clip_convnext_base(self.device)
        elif self.model_name == 'convnext_base_sup':
            return load_convnext_base(self.device)
        elif self.model_name == 'dinov2':
            return load_dinov2(self.device)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def get_features(self, x, patch_wise=False):
        assert x.min() >= 0 and x.max() <= 1, f'Input range is not [0,1]. Min: {x.min()}, Max: {x.max()}'
        x = self.transforms(x.to(torch.float32).clip(0,1))
        if self.model_name == 'inception':
            return self.f_model(x, return_features=True)
        elif self.model_name == 'vit_base_dino':
            return self.f_model(x)
        elif self.model_name == 'convnext_base_clip':
            return self.f_model.encode_image(x)  # [bs, 640]
        elif self.model_name == 'vit_base_clip':
            return self.f_model.encode_image(x)  # [bs, 640]
        elif self.model_name == 'convnext_base_sup':
            return self.f_model.forward_features(x)  # [bs, 1024]
        elif self.model_name == 'vit_base_sup':
            return self.f_model(x)
        elif self.model_name == 'dinov2':
            if not patch_wise:
                return self.f_model(x)
            else:
                with torch.no_grad():
                    output = self.f_model.get_intermediate_layers(x, n=1, reshape=False)[0]
                return output  # [bs, n_patches, 1024]
        else:
            raise NotImplementedError
