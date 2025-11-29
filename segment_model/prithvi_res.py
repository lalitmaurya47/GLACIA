
import torch
import torch.nn as nn
import torch.nn.functional as F

from terratorch.registry import BACKBONE_REGISTRY
# ----------------------------- 
#   BasicBlock (ResNet-18/34)
# ----------------------------- 



def convert_batchnorm_to_float(module):
    """Keep all BatchNorm parameters & buffers in float32."""
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        module.float()
        if module.weight is not None:
            module.weight.data = module.weight.data.float()
        if module.bias is not None:
            module.bias.data = module.bias.data.float()
        if hasattr(module, "running_mean") and module.running_mean is not None:
            module.running_mean = module.running_mean.float()
        if hasattr(module, "running_var") and module.running_var is not None:
            module.running_var = module.running_var.float()
    return module


def convert_module_to_half_safe(module, dtype=torch.float16):
    """
    Safe half/bf16 conversion:
      - Conv/Linear/etc → dtype
      - BatchNorm → float32
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            convert_batchnorm_to_float(m)
        else:
            # convert parameters to half
            for p in m.parameters(recurse=False):
                p.data = p.data.to(dtype)
            # convert buffers to half
            for b in m.buffers():
                try:
                    b.data = b.data.to(dtype)
                except Exception:
                    pass
    return module

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x # Store the original input
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity) # Apply shortcut to original input
        return F.relu(out)


# -----------------------------
#   Bottleneck (ResNet-50/101)
# ----------------------------- 
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


# -----------------------------
#   6-Channel ResNet Stem
# ----------------------------- 
class ResNetStem6(nn.Module):
    """
    Produces hierarchical CNN features (C2, C3, C4, C5)
    for 6-channel images, without pretrained parameters.
    """

    def __init__(self, block, layers, in_channels=6):
        super().__init__()

        self.in_planes = 64

        # ---- Modified first conv: 6-channel instead of 3-channel ----
        self.conv1 = nn.Conv2d(in_channels, 64,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- ResNet layers ----
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)  # C2 (/4)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # C3 (/8)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # C4 (/16)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # C5 (/32)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init (similar to default PyTorch init for resnet)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: B x 6 x H x W
        returns [C2, C3, C4, C5] feature maps.
        """
       
        x = self.conv1(x)   # /2
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)  # /4

        c2 = self.layer1(x)  # /4
        c3 = self.layer2(c2) # /8
        c4 = self.layer3(c3) # /16
        c5 = self.layer4(c4) # /32

        return [c2, c3, c4, c5]






class HybridPrithviEncoder(nn.Module):
    """
    Hybrid encoder: ResNet stem + Prithvi transformer features fused.
    - selected_layers: indices into transformer's out list (e.g., (2,5,8,11))
    - resnet_to_vit_map: which ResNet level to fuse for each selected layer
      (list/tuple same length as selected_layers), values in {0,1,2,3} mapping to [c2,c3,c4,c5]
    """
    def __init__(self,
                 model_name="prithvi_eo_v2_100_tl",
                 checkpoint_path="/users/lalit47/PRS-Med-main/weight/prithvi/Prithvi_EO_V2_100M_TL.pt",
                 selected_layers=(2, 5, 8, 11),
                 embed_dim=768,
                 out_channels=256,
                 tower_out_channels=1024,
                 use_tower=False,
                 resnet_variant="resnet34",
                 resnet_pretrained=True,
                 resnet_to_vit_map=(0, 1, 2, 3),
                 fusion_mode="concat"  # 'add' or 'concat'
                 ):
        super().__init__()

        # Transformer backbone (Prithvi)
        self.model = BACKBONE_REGISTRY.build(
            model_name,
            num_frames=1,
            ckpt_path=checkpoint_path,
        )
        self.selected_layers = tuple(selected_layers)
        self.use_tower = use_tower
        self.fusion_mode = fusion_mode

        if self.use_tower:
            out_channels = tower_out_channels

        # Example: ResNet-34 style for 6-channel input
        self.resnet = ResNetStem6(
            block=BasicBlock,
            layers=[3, 4, 6, 3],   # ResNet-34
            in_channels=6
        )
        
        
        

        if len(resnet_to_vit_map) != len(self.selected_layers):
            raise ValueError("resnet_to_vit_map length must match selected_layers length")
        self.resnet_to_vit_map = tuple(resnet_to_vit_map)

        # Project CNN channels (c2..c5) to out_channels for fusion
        # ResNet50 channels are [256, 512, 1024, 2048]
        # We'll create per-level projectors to handle arbitrary variants.
        res_channels_map = {
            "resnet50": [256, 512, 1024, 2048],
            "resnet34": [64, 128, 256, 512],
        }
        res_chs = res_channels_map.get(resnet_variant, [256, 512, 1024, 2048])

        # projection per resnet level (c2,c3,c4,c5 -> out_channels)
        self.res_projectors = nn.ModuleList([
            nn.Conv2d(res_ch, out_channels, kernel_size=1) for res_ch in res_chs
        ])

        # 1x1 lateral for transformer hidden -> out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, out_channels, 1) for _ in self.selected_layers
        ])

        # If fusion_mode == concat, we need a conv to reduce channels after concat
        if fusion_mode == "concat":
            self.fuse_reducer = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

        # Residual blocks (simple)
        def make_res_blocks(n=2):
            layers = []
            for _ in range(n):
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.res_blocks = nn.ModuleList([
            make_res_blocks() for _ in self.selected_layers
        ])

        # FPN smoothing convs
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, 1, 1) for _ in self.selected_layers
        ])

    def convert_to_half(self, dtype=torch.float16):
        """
        Convert entire encoder to half or bf16 safely:
          - Conv/Linear/Transformer → half
          - BatchNorm → float32
        """
        convert_module_to_half_safe(self, dtype=dtype)

    def forward(self, x):
        """
        x: Bx6xHxW
        All operations are dtype-safe for half precision.
        """
    
        # Ensure input matches model dtype
        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)
    
        # --------------------------
        # 1) ResNet features
        # --------------------------
        res_feats = self.resnet(x)
    
        # --------------------------
        # 2) Transformer features
        # --------------------------
        vit_out = self.model(x)
        hidden_states = []
        for i in self.selected_layers:
            tokens = vit_out[i][:, 1:, :]
            B, N, C = tokens.shape
            H = W = int(N ** 0.5)
            feat = tokens.transpose(1, 2).reshape(B, C, H, W)
            hidden_states.append(feat.to(model_dtype))
    
        # --------------------------
        # 3) Fuse CNN & ViT
        # --------------------------
        fused_feats = []
        for idx, (h_feat, lat_conv, res_block) in enumerate(
            zip(hidden_states, self.lateral_convs, self.res_blocks)
        ):
            # project transformer
            t = lat_conv(h_feat.to(model_dtype))
    
            # pick resnet level
            res_idx = self.resnet_to_vit_map[idx]
            r = res_feats[res_idx]
    
            # project CNN
            r_proj = self.res_projectors[res_idx](r.to(model_dtype))
    
            # resize CNN
            Ht, Wt = t.shape[-2], t.shape[-1]
            r_resized = F.interpolate(r_proj, size=(Ht, Wt), mode="bilinear",
                                      align_corners=False)
    
            # fuse
            if self.fusion_mode == "add":
                fused = t + r_resized
            else:
                fused = torch.cat([t, r_resized], dim=1)
                fused = self.fuse_reducer(fused)
    
            # residual blocks
            fused = res_block(fused)
            fused_feats.append(fused)
    
        # --------------------------
        # 4) FPN top-down fusion
        # --------------------------
        for i in range(len(fused_feats) - 1, 0, -1):
            up = F.interpolate(
                fused_feats[i],
                size=fused_feats[i - 1].shape[-2:],
                mode="nearest"
            )
            fused_feats[i - 1] = fused_feats[i - 1] + up
    
        # --------------------------
        # 5) Final outputs
        # --------------------------
        outs = [
            fpn(f.to(model_dtype))
            for fpn, f in zip(self.fpn_convs, fused_feats)
        ]
    
        final_out = outs[0]
    
        if self.use_tower:
            B, C, H, W = final_out.shape
            final_out = final_out.flatten(2).transpose(1, 2)
    
        return final_out.to(model_dtype)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from terratorch.registry import BACKBONE_REGISTRY
# # -----------------------------
# #   BasicBlock (ResNet-18/34)
# # -----------------------------
# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)

#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1,
#                           stride=stride, bias=False),
#                 nn.BatchNorm2d(planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(out)
#         return F.relu(out)


# # -----------------------------
# #   Bottleneck (ResNet-50/101)
# # -----------------------------
# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)

#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.conv3 = nn.Conv2d(planes, planes * self.expansion,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes * self.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes * self.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * self.expansion)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         return F.relu(out)


# # -----------------------------
# #   6-Channel ResNet Stem
# # -----------------------------
# class ResNetStem6(nn.Module):
#     """
#     Produces hierarchical CNN features (C2, C3, C4, C5)
#     for 6-channel images, without pretrained parameters.
#     """

#     def __init__(self, block, layers, in_channels=6):
#         super().__init__()

#         self.in_planes = 64

#         # ---- Modified first conv: 6-channel instead of 3-channel ----
#         self.conv1 = nn.Conv2d(in_channels, 64,
#                                kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)

#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         # ---- ResNet layers ----
#         self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)  # C2 (/4)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # C3 (/8)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # C4 (/16)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # C5 (/32)

#         self._init_weights()

#     def _init_weights(self):
#         """Kaiming init (similar to default PyTorch init for resnet)."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out",
#                                         nonlinearity="relu")
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, stride):
#         layers = [block(self.in_planes, planes, stride)]
#         self.in_planes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_planes, planes))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         """
#         x: B x 6 x H x W
#         returns [C2, C3, C4, C5] feature maps.
#         """
#         x = self.conv1(x)   # /2
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.maxpool(x)  # /4

#         c2 = self.layer1(x)  # /4
#         c3 = self.layer2(c2) # /8
#         c4 = self.layer3(c3) # /16
#         c5 = self.layer4(c4) # /32

#         return [c2, c3, c4, c5]







# class HybridPrithviEncoder(nn.Module):
#     """
#     Hybrid encoder: ResNet stem + Prithvi transformer features fused.
#     - selected_layers: indices into transformer's out list (e.g., (2,5,8,11))
#     - resnet_to_vit_map: which ResNet level to fuse for each selected layer
#       (list/tuple same length as selected_layers), values in {0,1,2,3} mapping to [c2,c3,c4,c5]
#     """
#     def __init__(self,
#                  model_name="prithvi_eo_v2_100_tl",
#                  checkpoint_path="/users/lalit47/PRS-Med-main/weight/prithvi/Prithvi_EO_V2_100M_TL.pt",
#                  selected_layers=(2, 5, 8, 11),
#                  embed_dim=768,
#                  out_channels=256,
#                  tower_out_channels=1024,
#                  use_tower=False,
#                  resnet_variant="resnet34",
#                  resnet_pretrained=True,
#                  resnet_to_vit_map=(0, 1, 2, 3),
#                  fusion_mode="concat"  # 'add' or 'concat'
#                  ):
#         super().__init__()

#         # Transformer backbone (Prithvi)
#         self.model = BACKBONE_REGISTRY.build(
#             model_name,
#             num_frames=1,
#             ckpt_path=checkpoint_path,
#         )
#         self.selected_layers = tuple(selected_layers)
#         self.use_tower = use_tower
#         self.fusion_mode = fusion_mode

#         if self.use_tower:
#             out_channels = tower_out_channels

#         # Example: ResNet-34 style for 6-channel input
#         self.resnet = ResNetStem6(
#             block=BasicBlock,
#             layers=[3, 4, 6, 3],   # ResNet-34
#             in_channels=6
#         )

#         if len(resnet_to_vit_map) != len(self.selected_layers):
#             raise ValueError("resnet_to_vit_map length must match selected_layers length")
#         self.resnet_to_vit_map = tuple(resnet_to_vit_map)

#         # Project CNN channels (c2..c5) to out_channels for fusion
#         # ResNet50 channels are [256, 512, 1024, 2048] for c2..c5
#         # We'll create per-level projectors to handle arbitrary variants.
#         res_channels_map = {
#             "resnet50": [256, 512, 1024, 2048],
#             "resnet34": [64, 128, 256, 512],
#         }
#         res_chs = res_channels_map.get(resnet_variant, [256, 512, 1024, 2048])

#         # projection per resnet level (c2,c3,c4,c5 -> out_channels)
#         self.res_projectors = nn.ModuleList([
#             nn.Conv2d(res_ch, out_channels, kernel_size=1) for res_ch in res_chs
#         ])

#         # 1x1 lateral for transformer hidden -> out_channels
#         self.lateral_convs = nn.ModuleList([
#             nn.Conv2d(embed_dim, out_channels, 1) for _ in self.selected_layers
#         ])

#         # If fusion_mode == concat, we need a conv to reduce channels after concat
#         if fusion_mode == "concat":
#             self.fuse_reducer = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

#         # Residual blocks (simple)
#         def make_res_blocks(n=2):
#             layers = []
#             for _ in range(n):
#                 layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
#                 layers.append(nn.BatchNorm2d(out_channels))
#                 layers.append(nn.ReLU(inplace=True))
#             return nn.Sequential(*layers)

#         self.res_blocks = nn.ModuleList([
#             make_res_blocks() for _ in self.selected_layers
#         ])

#         # FPN smoothing convs
#         self.fpn_convs = nn.ModuleList([
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1) for _ in self.selected_layers
#         ])

#     def forward(self, x):
#         """
#         x: Bx3xH xW
#         returns fused final_out (either spatial BxCxHxW or flattened tokens if use_tower)
#         """
#         # 1) ResNet stem features
#         res_feats = self.resnet(x)  # list: [c2,c3,c4,c5]

#         # 2) Transformer features (list of hidden states / tokens)
#         vit_out = self.model(x)  # assuming model returns list-like hidden states
#         hidden_states = []
#         for i in self.selected_layers:
#             tokens = vit_out[i][:, 1:, :]  # remove CLS token
#             B, N, C = tokens.shape
#             H = W = int(N ** 0.5)
#             feat = tokens.transpose(1, 2).reshape(B, C, H, W)  # BxCxHxW
#             hidden_states.append(feat)

#         # 3) lateral projection and fusion with corresponding resnet level
#         fused_feats = []
#         for idx, (h_feat, lat_conv, res_block) in enumerate(zip(hidden_states, self.lateral_convs, self.res_blocks)):
#             # project transformer tokens to out_channels
#             t = lat_conv(h_feat)  # B x out_ch x Ht x Wt

#             # choose which resnet level to fuse
#             res_idx = self.resnet_to_vit_map[idx]  # 0..3
#             r = res_feats[res_idx]  # B x Rc x Hr x Wr
#             # project resnet channels to out_channels
#             r_proj = self.res_projectors[res_idx](r)  # B x out_ch x Hr x Wr

#             # resize cnn feature to transformer grid
#             Ht, Wt = t.shape[-2], t.shape[-1]
#             r_resized = F.interpolate(r_proj, size=(Ht, Wt), mode="bilinear", align_corners=False)

#             # fuse
#             if self.fusion_mode == "add":
#                 fused = t + r_resized
#             elif self.fusion_mode == "concat":
#                 fused = torch.cat([t, r_resized], dim=1)  # B x (2*out_ch) x H x W
#                 fused = self.fuse_reducer(fused)
#             else:
#                 raise ValueError("unknown fusion_mode")

#             # local non-linearity / lightweight residual blocks
#             fused = res_block(fused)
#             fused_feats.append(fused)

#         # 4) Top-down FPN fusion (same as earlier)
#         for i in range(len(fused_feats) - 1, 0, -1):
#             up = F.interpolate(fused_feats[i], size=fused_feats[i - 1].shape[-2:], mode="nearest")
#             fused_feats[i - 1] = fused_feats[i - 1] + up

#         outs = [fpn(f) for fpn, f in zip(self.fpn_convs, fused_feats)]
#         final_out = outs[0]

#         if self.use_tower:
#             B, C, H, W = final_out.shape
#             final_out = final_out.flatten(2).transpose(1, 2)  # B x (H*W) x C

#         return final_out
