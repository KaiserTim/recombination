import torch
import dnnlib
import torch.nn as nn

from torch_utils import misc

def load_swav():
    torch.hub.set_dir(dnnlib.make_cache_dir_path('torch_hub'))
    full_model = torch.hub.load('facebookresearch/swav:main', 'resnet50', trust_repo=True, verbose=False)
    f_dim = 2048
    f_model = torch.nn.Sequential(*list(full_model.children())[:-1])  # drop classification head
    return f_model, f_dim


def load_dinov2():
    import warnings
    warnings.filterwarnings('ignore', 'xFormers is not available')
    torch.hub.set_dir(dnnlib.make_cache_dir_path('torch_hub'))
    f_dim = 1024
    f_model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitl14', trust_repo=True, verbose=False)
    return f_model, f_dim


class DinoPatchMeanWrapper(nn.Module):
    """DINOv2 wrapper to extract patch means instead of aggregating them in the CLS token."""
    def __init__(self, dinov2_model):
        super().__init__()
        self.model = dinov2_model

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1  # exclude CLS
        N = self.model.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.model.pos_embed
        class_pos_embed = self.model.pos_embed[:, 0]  # (bs, 1, D)
        patch_pos_embed = self.model.pos_embed[:, 1:]  # (bs, N, D)
        dim = x.shape[-1]
        w0 = w // self.model.patch_embed.patch_size[0]
        h0 = h // self.model.patch_embed.patch_size[0]
        patch_pos_embed = patch_pos_embed.reshape(1, int(N ** 0.5), int(N ** 0.5), dim)  # (bs, sqrt(N), sqrt(N), D
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # (bs, D, sqrt(N), sqrt(N))
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed, size=(h0, w0), mode='bicubic', align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat((class_pos_embed.unsqueeze(1), patch_pos_embed), dim=1)  # (1, new_N, D)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.model.patch_embed(x)  # (B, N, D)
        cls_token = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        pos_embed = self.interpolate_pos_encoding(x, W, H)
        x = x + pos_embed
        if hasattr(self.model, 'pos_drop'):
            x = self.model.pos_drop(x)

        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)

        patch_tokens = x[:, 1:, :]  # remove CLS
        patch_mean = patch_tokens.mean(dim=1)
        return patch_mean


class FeatureExtractor:
    def __init__(self, model_name, device='cuda'):
        self.model_name = model_name
        self.device = device
        self.f_model, self.f_dim = self.load_feature_model()

    def load_feature_model(self):
        if self.model_name == 'dinov2':
            f_model, f_dim = load_dinov2()
            f_model = DinoPatchMeanWrapper(f_model).eval()
        elif self.model_name == 'swav':
            f_model, f_dim = load_swav()
        else:
            raise NotImplementedError

        f_model = f_model.to(self.device)
        f_model.eval().requires_grad_(False)
        return f_model, f_dim

    @torch.no_grad()
    def get_features(self, x):
        return self.f_model(x)  # [bs, f_dim]
