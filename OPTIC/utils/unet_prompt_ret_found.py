# Copyright 2023 Huawei Technologies Co., Ltd
# Modified from original SwinPrompt implementation

import logging
import math
from functools import partial

# import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_

from mmseg.models.backbones.unet import UNet
from .adapter_modules import InteractionBlockPrompt
from .ret_found import RetFound


@BACKBONES.register_module()
class UNetPromptRETBase(UNet):
    def __init__(
        self,
        freeze_backbone=True,
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU"),
        upsample_cfg=dict(type="InterpConv"),
        norm_eval=False,
        dcn=None,
        plugins=None,
        pretrained=None,
        init_cfg=None,
        drop_path_rate=0.1,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-6,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        norm_layer=None,
        use_extra_extractor=False,
    ):

        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            strides=strides,
            enc_num_convs=enc_num_convs,
            dec_num_convs=dec_num_convs,
            downsamples=downsamples,
            enc_dilations=enc_dilations,
            dec_dilations=dec_dilations,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            norm_eval=norm_eval,
            dcn=dcn,
            plugins=plugins,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # self.next_dim = [256, 512, 1024, None]
        self.next_dim = [base_channels * 2, base_channels * 4, base_channels * 8, None]
        # self.next_dim = [base_channels, base_channels * 2, base_channels * 4, None]
        self.spm1 = RetFound(img_size=256, embed_dim=64)
        self.spm2 = RetFound(img_size=512, embed_dim=64)
        self.spm3 = RetFound(img_size=1024, embed_dim=64)

        # Create interaction blocks for each stage
        self.interactions = nn.Sequential(
            *[
                InteractionBlockPrompt(
                    dim=base_channels * 2**i,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=use_extra_extractor,
                    with_cp=with_cp,
                    next_dim=self.next_dim[i],
                )
                for i in range(
                    num_stages - 1
                )  # -1 because UNet has one less decoder stage
            ]
        )

        self.spm1.load_pretrained_weights(
            "/root/autodl-tmp/uni-uvpt/model/RETFound/RETFound_cfp_weights.pth"
        )
        self.spm2.load_pretrained_weights(
            "/root/autodl-tmp/uni-uvpt/model/RETFound/RETFound_cfp_weights.pth"
        )
        self.spm3.load_pretrained_weights(
            "/root/autodl-tmp/uni-uvpt/model/RETFound/RETFound_cfp_weights.pth"
        )
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)

        if freeze_backbone:
            for name, params in self.named_parameters():
                if "spm" in name or "interactions" in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # SPM forward
        if x.shape[2] == 128:
            c1, c2, c3 = self.spm1(x)
        elif x.shape[2] == 256:
            c1, c2, c3 = self.spm2(x)
        elif x.shape[2] == 512:
            c1, c2, c3 = self.spm3(x)

        # Get UNet encoder features
        self._check_input_divisible(x)
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)

        # Get decoder features with prompt interaction
        x = enc_outs[-1]  # Start with bottleneck features
        dec_outs = [x]

        hw_shapes = []
        for feat in enc_outs:
            hw_shapes.append(feat.shape[2:])

        # Process through decoder stages with interactions
        for i in reversed(range(len(self.decoder))):
            # Get current stage's spatial dimensions
            h, w = hw_shapes[i]
            hw_shape = (h, w)

            # Interaction with prompts
            x_injector, c1, c2, c3, _, hw_shape, out, _ = self.interactions[i](
                enc_outs[i], c1, c2, c3, None, hw_shape, backbone="unet"
            )

            # UNet decoder processing
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)

        return dec_outs


@BACKBONES.register_module()
class UNetPromptRETFound(UNetPromptRETBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add learnable level embeddings
        self.level_embed = nn.Parameter(torch.zeros(3, kwargs.get("base_channels", 64)))
        normal_(self.level_embed)

        # Update freezing logic to include level embeddings
        if kwargs.get("freeze_backbone", True):
            for name, params in self.named_parameters():
                if "interactions" in name or "level_embed" in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False

    def _add_level_embed(self, c1, c2, c3):
        c1 = c1 + self.level_embed[0]
        c2 = c2 + self.level_embed[1]
        c3 = c3 + self.level_embed[2]
        return c1, c2, c3

    def forward(self, x):
        out_prompt = []
        # ipdb.set_trace()
        x_upsampled = F.interpolate(
            x,
            size=(x.shape[2] * 2, x.shape[2] * 2),
            mode="bilinear",
            align_corners=False,
        )
        if x.shape[2] == 128:
            c1, c2, c3 = self.spm1(x_upsampled)
        elif x.shape[2] == 256:
            c1, c2, c3 = self.spm2(x_upsampled)
        elif x.shape[2] == 512:
            c1, c2, c3 = self.spm3(x_upsampled)

        c1, c2, c3 = self._add_level_embed(c1, c2, c3)
        # Get UNet encoder features with prompt interaction
        enc_outs = []
        for i, enc in enumerate(self.encoder):
            # Perform prompt interaction at each encoder stage
            # if i < len(self.encoder) - 1:
            if i > 0:
                h, w = x.shape[2:]
                # ipdb.set_trace()
                hw_shape = (h, w)
                x = x.flatten(2).transpose(1, 2)
                x_injector, c1, c2, c3, out = self.interactions[i - 1](
                    x, c1, c2, c3, enc, hw_shape, backbone="UNet"
                )
                out_prompt.append(x_injector)
                x = out
                enc_outs.append(out)
            else:
                x = enc(x)
                x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
                enc_outs.append(x)

        # Process with decoder
        # x = enc_outs[-1]
        dec_outs = [x]

        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)

        outputs = dict()
        outputs["outs"] = dec_outs
        outputs["prompts"] = out_prompt
        return outputs
