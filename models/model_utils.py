import torch
import dnnlib

from torch_utils import misc

def load_swav(device='cuda'):
    torch.hub.set_dir(dnnlib.make_cache_dir_path('torch_hub'))
    full_model = torch.hub.load('facebookresearch/swav:main', 'resnet50', trust_repo=True, verbose=False)
    f_model = torch.nn.Sequential(*list(full_model.children())[:-1])  # drop classification head
    f_model = f_model.to(device)
    f_model.eval().requires_grad_(False)

    def transforms(x):
        x = torch.nn.functional.interpolate(x.to(torch.float32), size=(224, 224), mode='bicubic', antialias=True)
        x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        return x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    f_dim = 2048

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
        x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)  # Adjust dynamic range.
        return x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    f_dim = 1024

    return f_model, transforms, f_dim



class FeatureExtractor:
    def __init__(self, model_name, device='cuda'):
        self.model_name = model_name
        self.device = device
        self.f_model, self.transforms, self.f_dim = self.load_feature_model()

    def load_feature_model(self):
        if self.model_name == 'dinov2':
            return load_dinov2(self.device)
        elif self.model_name == 'swav':
            return load_swav(self.device)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def get_features(self, x):
        assert x.min() >= 0 and x.max() <= 1, f'Input range is not [0,1]. Min: {x.min()}, Max: {x.max()}'
        x = self.transforms(x.to(torch.float32).clip(0,1))
        if self.model_name == 'dinov2':
            return self.f_model.get_intermediate_layers(x, n=1, reshape=False)[0]  # [bs, n_patches, 1024]
        elif self.model_name == 'swav':
            return self.f_model(x)  # [bs, 2048]
        else:
            raise NotImplementedError
