import torch
import dnnlib
import torch.nn as nn
import torch.nn.functional as F
import lpips


# def compute_dists_distance(feats1, feats2):
#     """Compute DISTS similarity given extracted features.
#     This is meant to be used after NN search with FAISS.
#
#     Args:
#         feats1: Features from the first image, shape [N, D]
#         feats2: Features from the second image, shape [N, D]
#
#     Returns:
#         DISTS similarity score (0-1, lower is more similar)
#     """
#     # Split concatenated features back into means and stds
#     D = feats1.shape[1] // 2
#
#     # Extract means and stds
#     mu1, sigma1 = feats1[:, :D], feats1[:, D:]
#     mu2, sigma2 = feats2[:, :D], feats2[:, D:]
#
#     # Calculate structure similarity (correlation of means)
#     structure_sim = F.cosine_similarity(mu1, mu2, dim=1)
#
#     # Calculate texture similarity (correlation of standard deviations)
#     texture_sim = F.cosine_similarity(sigma1, sigma2, dim=1)
#
#     # Combine structure and texture similarities (using equal weights by default)
#     alpha = 0.5  # Balance between structure and texture
#     similarity = alpha * structure_sim + (1 - alpha) * texture_sim
#
#     # Convert to distance (0-1, lower is more similar)
#     distance = 1 - similarity
#
#     return distance.mean()


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


class LPIPSFeatures(nn.Module):
    """VGG16 or AlexNet feature extractor for LPIPS."""

    def __init__(self, net_type):
        super().__init__()
        self.lpips_model = lpips.LPIPS(net=net_type, pretrained=True)
        self.lpips_model.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.lpips_model.net.forward(x)

        all_weights = []
        for i, lin in enumerate(self.lpips_model.lins):
            all_weights.append(lin.model[-1].weight.data.squeeze())

        weighted_features = []
        for i, feat in enumerate(features):
            norm = feat.pow(2).sum(dim=1, keepdim=True).pow(0.5)
            normalized_feat = feat / (norm + 1e-10)  # Add small epsilon to avoid division by zero
            avg_feat = normalized_feat.mean(dim=(2,3))  # Apply global average pooling to each feature map, as we don't need spatial information
            weighted_feature = avg_feat * all_weights[i][None, :] ** 0.5  # Weight is now inside the L2-square
            weighted_features.append(weighted_feature)

        return torch.cat([f.flatten(start_dim=1) for f in weighted_features], dim=1)  # Create a single feature vector


# class DISTSFeatures(nn.Module):
#     """VGG16 feature extractor for DISTS.
#     Based on implementation from https://github.com/dingkeyan93/DISTS
#     """
#     def __init__(self):
#         super().__init__()
#         vgg_pretrained_features = models.vgg16(pretrained=True).features
#         self.stage1 = nn.Sequential()
#         self.stage2 = nn.Sequential()
#         self.stage3 = nn.Sequential()
#         self.stage4 = nn.Sequential()
#         self.stage5 = nn.Sequential()
#
#         for x in range(0, 4):
#             self.stage1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(4, 9):
#             self.stage2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(9, 16):
#             self.stage3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(16, 23):
#             self.stage4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(23, 30):
#             self.stage5.add_module(str(x), vgg_pretrained_features[x])
#
#         # Freeze parameters
#         for param in self.parameters():
#             param.requires_grad = False
#
#         # Pretrained coefficients for channel-wise normalization
#         self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
#         self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])
#
#     def forward(self, x):
#         # Normalize input
#         x = (x - self.shift) / self.scale
#
#         # Extract features from each layer
#         h1 = self.stage1(x)
#         h2 = self.stage2(h1)
#         h3 = self.stage3(h2)
#         h4 = self.stage4(h3)
#         h5 = self.stage5(h4)
#
#         # Calculate mean and standard deviation for each feature map
#         features = [h1, h2, h3, h4, h5]
#         feature_means = []
#         feature_stds = []
#
#         for feat in features:
#             # Mean and std across spatial dimensions
#             mean = feat.mean(dim=[2, 3])
#             std = feat.std(dim=[2, 3])
#
#             # Flatten for concatenation
#             feature_means.append(mean)
#             feature_stds.append(std)
#
#         # Concatenate all features
#         all_means = torch.cat(feature_means, dim=1)
#         all_stds = torch.cat(feature_stds, dim=1)
#
#         # For DISTS, we concatenate means and stds for distance computation later
#         combined_features = torch.cat([all_means, all_stds], dim=1)
#         return combined_features


class FeatureExtractor:
    def __init__(self, model_name, device='cuda'):
        """
        net_type (str): 'alex' or 'vgg' for LPIPS feature extraction.
        """
        self.model_name = model_name
        self.device = device

        if model_name == 'dinov2':
            f_model, f_dim = self.load_dinov2()
            f_model = DinoPatchMeanWrapper(f_model).eval()
        elif model_name == 'swav':
            f_model, f_dim = self.load_swav()
        elif 'lpips' in model_name:
            f_model, f_dim = self.load_lpips()
        elif model_name == 'dists':
            f_model, f_dim = self.load_dists()
        else:
            raise NotImplementedError(f"Feature model {model_name} not implemented")

        f_model = f_model.to(self.device)
        f_model.eval().requires_grad_(False)
        self.f_model = f_model
        self.f_dim = f_dim

    @staticmethod
    def load_swav():
        torch.hub.set_dir(dnnlib.make_cache_dir_path('torch_hub'))
        full_model = torch.hub.load('facebookresearch/swav:main', 'resnet50', trust_repo=True, verbose=False)
        f_dim = 2048
        f_model = torch.nn.Sequential(*list(full_model.children())[:-1])  # drop classification head
        return f_model, f_dim

    @staticmethod
    def load_dinov2():
        import warnings
        warnings.filterwarnings('ignore', 'xFormers is not available')
        torch.hub.set_dir(dnnlib.make_cache_dir_path('torch_hub'))
        f_dim = 1024
        f_model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitl14', trust_repo=True, verbose=False)
        return f_model, f_dim

    def load_lpips(self):
        """Load pretrained VGG LPIPS feature extraction.
        Based on original implementation from https://github.com/richzhang/PerceptualSimilarity
        """
        f_dim = None  # Depends on the input size
        model = LPIPSFeatures(net_type=self.model_name.split('-')[-1])
        return model, f_dim

    # @staticmethod
    # def load_dists():
    #     """Load pretrained VGG model for DISTS feature extraction.
    #     Based on implementation from https://github.com/dingkeyan93/DISTS
    #     """
    #     model = DISTSFeatures()
    #     f_dim = 45056  # Concatenated features dimension (mean + std features)
    #     return model, f_dim

    @torch.no_grad()
    def get_features(self, x):
        return self.f_model(x)  # [bs, f_dim]
