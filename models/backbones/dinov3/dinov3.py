import torch
import torch.nn as nn

from .dinov3_repo import DINOV3_REPO_PATH
from .urls import dinov3_vitb16


class DINOv3(nn.Module):
    def __init__(
            self,
            model_name: str,
            num_trainable_blocks: int = 2,
            norm_layer: bool = False,
            return_token: bool = False,
            rope_sincos: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = torch.hub.load(
            DINOV3_REPO_PATH,
            model_name,
            source = 'local',
            weights = dinov3_vitb16
        )
        self.num_channels = self.model.num_features
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token
        self.rope_sincos = rope_sincos

        if self.num_trainable_blocks > 0:
            self.frozen_blocks = self.model.blocks[:-self.num_trainable_blocks]
            self.trainable_blocks = self.model.blocks[-self.num_trainable_blocks:]
        else:
            self.frozen_blocks = self.model.blocks
            self.trainable_blocks = []
        
        self.freeze_blocks()

    def freeze_blocks(self) -> None:
        for blk in self.frozen_blocks:
            for param in blk.parameters():
                param.requires_grad = False
        
        if self.num_trainable_blocks == 0:
            for param in self.model.norm.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple:
        B, _, h, w = x.shape
        x, (H, W) = self.model.prepare_tokens_with_masks(x)

        if self.rope_sincos:
            rope_sincos = self.model.rope_embed(H=H, W=W)
        else:
            rope_sincos = None

        # First blocks are frozen
        with torch.no_grad():
            for blk in self.frozen_blocks:
                x = blk(x, rope_sincos)

        # Last blocks are trained
        for blk in self.trainable_blocks:
            x = blk(x, rope_sincos)

        if self.norm_layer:
            x = self.model.norm(x)

        class_token = x[:, 0]
        register_token = x[:, 1: self.model.n_storage_tokens + 1] #Probably it adds nothing for inference

        features = x[:, self.model.n_storage_tokens + 1 :]

        features = features.reshape((
            B,
            h // self.model.patch_size,
            w // self.model.patch_size,
            self.num_channels
        )).permute(0, 3, 1, 2).contiguous()

        if self.return_token:
            return features, class_token
        return features
