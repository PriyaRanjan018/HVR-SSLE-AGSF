import os
import sys
import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    from .swin2sr import Swin2SR_DFE
    from .LightUnetPP import LightUNetPlusPlus
    from .utils import pad_to_multiple_calculator, pad_to_multiple, crop_to_original, CnnEmbeddingLayer, lcm_of_list
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    from models.swin2sr import Swin2SR_DFE
    from models.LightUnetPP import LightUNetPlusPlus
    from models.utils import pad_to_multiple_calculator, pad_to_multiple, crop_to_original, CnnEmbeddingLayer, lcm_of_list


class AdaptiveGatedFusion(nn.Module):
    """
    AGSF: Adaptive Gated State Fusion
    Replaces plain (zL + zH) addition with a learned per-channel gate.
    gate  = sigmoid( Conv1x1( concat(zL, zH) ) )
    fused = gate * zL  +  (1 - gate) * zH
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        # Init gate bias to 0 so sigmoid(0)=0.5 at start (equal mix)
        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, zL: torch.Tensor, zH: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([zL, zH], dim=1)))
        return g * zL + (1.0 - g) * zH


class HVR_AGSF(nn.Module):
    def __init__(self, config, checkpoint: bool = True):
        super(HVR_AGSF, self).__init__()
        self.config = config
        self.checkpoint = checkpoint
        self.T = config.HVR_T
        self.C = config.HVR_C
        self.N_supervision = config.HVR_N_supervision

        self.image_input_channels = 3
        self.embed_layer = CnnEmbeddingLayer(
            in_channels=self.image_input_channels,
            output_channels=config.embed_dims,
        )

        self.L_net_model = LightUNetPlusPlus(
            in_channels=config.embed_dims,
            z_channels=config.z_dims,
            out_channels=config.z_dims,
            init_features=config.LightUnetPP.get("init_features", 64),
            num_groups=config.LightUnetPP.get("num_groups", 16),
        )

        input_image_h = config.height
        input_image_w = config.width
        self.padding_multiply_factor = lcm_of_list(
            [self.L_net_model.padding_multiply_factor, config.Swin2SR.get("window_size", 8)]
        )
        (_, _), (h_padded, w_padded) = pad_to_multiple_calculator(
            h=input_image_h, w=input_image_w, multiple=self.padding_multiply_factor
        )

        self.H_net_model = Swin2SR_DFE(
            img_size=(h_padded, w_padded),
            patch_size=1,
            embed_dim=config.z_dims,
            depths=config.Swin2SR.get("depths", [3, 3, 3]),
            num_heads=config.Swin2SR.get("num_heads", [4, 4, 4]),
            window_size=config.Swin2SR.get("window_size", 8),
            mlp_ratio=config.Swin2SR.get("mlp_ratio", 4.0),
            qkv_bias=True,
            drop_rate=config.Swin2SR.get("drop_rate", 0.1),
            attn_drop_rate=config.Swin2SR.get("attn_drop_rate", 0.1),
            drop_path_rate=config.Swin2SR.get("drop_path_rate", 0.1),
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=self.checkpoint,
            img_range=1.0,
            resi_connection=config.Swin2SR.get("resi_connection", "1conv"),
        )

        self.head_layer = CnnEmbeddingLayer(
            in_channels=config.z_dims,
            output_channels=self.image_input_channels,
            enable_activation=True,
            last_one_to_one=True if config.normalize_one_to_one else False,
        )

        # ── AGSF modules (the only addition vs baseline) ──
        self.L_fusion = AdaptiveGatedFusion(config.z_dims)  # for L_net input
        self.H_fusion = AdaptiveGatedFusion(config.z_dims)  # for H_net input

    # ── Modified L_net: uses AGSF instead of zL + zH ──
    def L_net(self, x_embed, zL, zH):
        fused = self.L_fusion(zL, zH)          # adaptive gate replaces zL + zH
        zL_out = self.L_net_model(x_embed, fused)
        return zL + zL_out

    # ── Modified H_net: uses AGSF instead of zH + zL ──
    def H_net(self, zH, zL):
        fused = self.H_fusion(zH, zL)          # adaptive gate replaces zH + zL
        return self.H_net_model(fused)

    @classmethod
    def z_init(cls, shape, zeros=False, a=-1.0, b=1.0, mean=0.0, std=1.0, device=None, dtype=None):
        result = torch.zeros(shape, device=device, dtype=dtype)
        if zeros:
            return result
        torch.nn.init.trunc_normal_(result, a=a, b=b, mean=mean, std=std)
        return result

    def forward(self, x, z: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        batch_size, _, h, w = x.shape
        x_padded_org, (_, _) = pad_to_multiple(x, multiple=self.padding_multiply_factor)

        if z is None:
            zH = self.z_init(shape=(batch_size, self.config.z_dims, h, w),
                             device=x.device, dtype=x.dtype, **self.config.z_distribution)
            zL = self.z_init(shape=(batch_size, self.config.z_dims, h, w),
                             device=x.device, dtype=x.dtype, **self.config.z_distribution)
            z = (zH, zL)

        zH, zL = z
        zH, (_, _) = pad_to_multiple(zH, multiple=self.padding_multiply_factor)
        zL, (_, _) = pad_to_multiple(zL, multiple=self.padding_multiply_factor)

        with torch.no_grad():
            x_embed = self.embed_layer(x_padded_org)
            for _i in range(self.T * self.C - 1):
                zL = self.L_net(x_embed, zL, zH)
                if (_i + 1) % self.T == 0:
                    zH = self.H_net(zH, zL)

        zH = zH.detach()
        zL = zL.detach()

        x_embed = self.embed_layer(x_padded_org)
        zL = self.L_net(x_embed, zL, zH)
        zH = self.H_net(zH, zL)

        output = self.head_layer(zH)
        output = crop_to_original(output, (h, w))
        zH = crop_to_original(zH, (h, w))
        zL = crop_to_original(zL, (h, w))

        return (zH, zL), output

    @torch.inference_mode()
    def sample(self, x, z=None, T=None, C=None, N_supervision=None, padding_multiply_factor=None):
        T = self.T if T is None else T
        C = self.C if C is None else C
        N_supervision = self.N_supervision if N_supervision is None else N_supervision
        if padding_multiply_factor is None:
            padding_multiply_factor = self.padding_multiply_factor

        batch_size, _, h, w = x.shape
        if z is None:
            zH = self.z_init(shape=(batch_size, self.config.z_dims, h, w),
                             device=x.device, dtype=x.dtype, **self.config.z_distribution)
            zL = self.z_init(shape=(batch_size, self.config.z_dims, h, w),
                             device=x.device, dtype=x.dtype, **self.config.z_distribution)
            z = (zH, zL)

        zH, zL = z
        zH, (_, _) = pad_to_multiple(zH, multiple=padding_multiply_factor)
        zL, (_, _) = pad_to_multiple(zL, multiple=padding_multiply_factor)
        x_padded, (_, _) = pad_to_multiple(x, multiple=padding_multiply_factor)
        x_embed = self.embed_layer(x_padded)

        for _s in range(N_supervision):
            for _i in range(T * C):
                zL = self.L_net(x_embed, zL, zH)
                if (_i + 1) % T == 0:
                    zH = self.H_net(zH, zL)

        output = self.head_layer(zH)
        output = crop_to_original(output, (h, w))
        zH = crop_to_original(zH, (h, w))
        zL = crop_to_original(zL, (h, w))
        return (zH, zL), output
