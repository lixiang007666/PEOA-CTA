import torch
import torch.nn as nn
import torch.nn.functional as F
from . import models_vit
from .pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_


class RetFound(nn.Module):
    def __init__(self, img_size=256, embed_dim=128):
        super(RetFound, self).__init__()
        vit_model = models_vit.__dict__["vit_large_patch16"](
            img_size=img_size,
            num_classes=3,
            drop_path_rate=0.2,
            global_pool=True,
        )
        self.patch_embed = vit_model.patch_embed
        self.pos_embed = vit_model.pos_embed  # 添加 pos_embed
        self.pos_drop = vit_model.pos_drop
        self.norm_pre = vit_model.norm_pre
        self.blocks = vit_model.blocks
        self.fc = nn.Conv2d(
            1024, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed[:, :1, :])  # 正确使用 pos_embed
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            with torch.no_grad():
                x = blk(x)

        B, N, C = x.shape
        c1 = (
            self.fc(
                F.interpolate(
                    x.permute(0, 2, 1).view(B, C, int(N**0.5), int(N**0.5)),
                    scale_factor=8,
                    mode="bilinear",
                )
            )
            .flatten(2)
            .permute(0, 2, 1)
        )
        c2 = (
            self.fc(
                F.interpolate(
                    x.permute(0, 2, 1).view(B, C, int(N**0.5), int(N**0.5)),
                    scale_factor=4,
                    mode="bilinear",
                )
            )
            .flatten(2)
            .permute(0, 2, 1)
        )
        c3 = (
            self.fc(
                F.interpolate(
                    x.permute(0, 2, 1).view(B, C, int(N**0.5), int(N**0.5)),
                    scale_factor=2,
                    mode="bilinear",
                )
            )
            .flatten(2)
            .permute(0, 2, 1)
        )

        return c1, c2, c3

    # def load_pretrained_weights(self, weights_path='../../../model/RETFound/RETFound_cfp_weights.pth'):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     checkpoint = torch.load(weights_path, map_location=device)
    #     checkpoint_model = checkpoint['model']
    #     state_dict = self.state_dict()
    #     for k in ['head.weight', 'head.bias']:
    #         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]
    #
    #     interpolate_pos_embed(self, checkpoint_model)  # 确保模型包含 pos_embed
    #     msg = self.load_state_dict(checkpoint_model, strict=False)
    #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
    #     trunc_normal_(self.patch_embed.proj.weight, std=2e-5)

    # interpolate_pos_embed(self, checkpoint_model)
    # msg = self.load_state_dict(checkpoint_model, strict=False)
    # # 打印缺失的键，帮助定位问题
    # print(f"Missing keys: {msg.missing_keys}")
    # print(f"Unexpected keys: {msg.unexpected_keys}")
    #
    # # 临时移除断言，允许检查缺失键
    # trunc_normal_(self.patch_embed.proj.weight, std=2e-5)
    def load_pretrained_weights(
        self, weights_path="../../../model/RETFound/RETFound_cfp_weights.pth"
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weights_path, map_location=device)
        checkpoint_model = checkpoint["model"]
        state_dict = self.state_dict()

        # 移除不需要的 decoder 相关的权重和其他不匹配的权重
        unwanted_keys = [
            "cls_token",
            "mask_token",
            "decoder_pos_embed",
            "norm.weight",
            "norm.bias",
            "decoder_embed.weight",
            "decoder_embed.bias",
        ]
        unwanted_keys += [
            k
            for k in checkpoint_model.keys()
            if k.startswith("decoder_blocks") or k.startswith("decoder_")
        ]

        for k in unwanted_keys:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # 忽略不匹配的 head、fc 层
        for k in ["head.weight", "head.bias", "fc.weight", "fc.bias"]:
            if (
                k in checkpoint_model
                and k in state_dict
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(
                    f"Removing key {k} from pretrained checkpoint due to size mismatch"
                )
                del checkpoint_model[k]

        # 执行插值嵌入并加载模型权重
        interpolate_pos_embed(self, checkpoint_model)
        msg = self.load_state_dict(checkpoint_model, strict=False)

        # 打印信息以确认哪些键未加载
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")

        # 初始化 fc 层权重
        if "fc.weight" in msg.missing_keys:
            print("Initializing fc layer weights")
            trunc_normal_(self.fc.weight, std=2e-5)
        if "fc.bias" in msg.missing_keys:
            self.fc.bias.data.zero_()
