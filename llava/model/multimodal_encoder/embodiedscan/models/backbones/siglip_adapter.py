import torch
from torch import nn
import torch.nn.functional as F
import copy

from llava.model.multimodal_encoder.embodiedscan.models.backbones.siglip_encoder import SigLipVisionTower, SigLipVisionModel, SigLipVisionConfig
from mmengine.model import BaseModel
from llava.model.multimodal_encoder.embodiedscan.registry import MODELS

class HierarchicalTransformerAggregation(nn.Module):
    """
    Hierarchical transformer aggregation that supports per-scale output channels.

    Input: x_merge (B*V, C_in, H_in, W_in)
    scales: list of (H,W) for outputs, processed from high->low or low->low depending on scales.
            If the input spatial size differs from scales[0], the module will resample (avg pool if integer factor, else bilinear)
            to match scales[0] before processing.
    out_channels_per_scale: list of ints, len == len(scales), defines output channels for each scale.
    """
    def __init__(
        self,
        in_ch,
        out_channels_per_scale,
        d_model=512,
        nhead=8,
        layers_per_scale=2,
        scales=[(48,48),(24,24),(12,12)],
        dim_feedforward=None,
        dropout=0.0,
        use_pos_emb=True,
    ):
        super().__init__()
        assert len(scales) == len(out_channels_per_scale), "scales and out_channels_per_scale must match length"
        assert layers_per_scale >= 1

        self.in_ch = in_ch
        self.scales = list(scales)
        self.out_channels_per_scale = list(out_channels_per_scale)
        self.d_model = d_model
        self.nhead = nhead
        self.layers_per_scale = layers_per_scale
        self.use_pos_emb = use_pos_emb

        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)

        # per-scale transformer encoders
        self.scale_encoders = nn.ModuleList()
        for _ in self.scales:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            enc = nn.TransformerEncoder(layer, num_layers=layers_per_scale)
            self.scale_encoders.append(enc)

        # shared projection in (channels -> d_model)
        # self.proj_in = nn.Linear(self.in_ch, self.d_model)
        self.in_chs_for_proj = [in_ch] + self.out_channels_per_scale[:-1]  # first: original in_ch, then previous out channels
        self.proj_ins = nn.ModuleList()
        for ch in self.in_chs_for_proj:
            self.proj_ins.append(nn.Conv2d(ch, self.d_model, kernel_size=1))

        # per-scale projection out (d_model -> out_ch_i)
        self.proj_outs = nn.ModuleList()
        for out_ch in self.out_channels_per_scale:
            self.proj_outs.append(nn.Linear(self.d_model, out_ch))

        # pos emb per scale (length = H_s * W_s)
        self.pos_embs = nn.ParameterList()
        for (H_s, W_s) in self.scales:
            seq_len = H_s * W_s
            if self.use_pos_emb:
                p = nn.Parameter(torch.zeros(1, seq_len, self.d_model))
                nn.init.normal_(p, std=0.02)
            else:
                p = nn.Parameter(torch.zeros(1, 0, self.d_model), requires_grad=False)
            self.pos_embs.append(p)

    def forward(self, x_merge):
        # x_merge: (B*V, C_in, H_in, W_in)
        BV, C_in, H_in, W_in = x_merge.shape
        device = x_merge.device
        feats = []

        # target first scale
        H0, W0 = self.scales[0]

        # If input size != first scale, downsample/up to match first scale.
        if (H_in, W_in) != (H0, W0):
            # try avg pool when integer downsample factor exists
            factor_h = H_in // H0 if H0 > 0 else None
            factor_w = W_in // W0 if W0 > 0 else None
            if (factor_h is not None and factor_w is not None and
                factor_h >= 1 and factor_w >= 1 and (H_in % H0 == 0) and (W_in % W0 == 0)):
                if factor_h == 1 and factor_w == 1:
                    cur = x_merge
                else:
                    cur = F.avg_pool2d(x_merge, kernel_size=(factor_h, factor_w), stride=(factor_h, factor_w))
            else:
                cur = F.interpolate(x_merge, size=(H0, W0), mode='bilinear', align_corners=False)
        else:
            cur = x_merge

        # sequentially process each scale
        for i, (H_s, W_s) in enumerate(self.scales):
            # ensure cur spatial matches H_s,W_s
            if cur.shape[-2:] != (H_s, W_s):
                # try avg pool first when integer factor, else interpolate
                h, w = cur.shape[-2], cur.shape[-1]
                factor_h = h // H_s if H_s > 0 else None
                factor_w = w // W_s if W_s > 0 else None
                if (factor_h is not None and factor_w is not None and
                    factor_h >= 1 and factor_w >= 1 and (h % H_s == 0) and (w % W_s == 0)):
                    if factor_h == 1 and factor_w == 1:
                        pass
                    else:
                        cur = F.avg_pool2d(cur, kernel_size=(factor_h, factor_w), stride=(factor_h, factor_w))
                else:
                    cur = F.interpolate(cur, size=(H_s, W_s), mode='bilinear', align_corners=False)

            BV2, Cc, h, w = cur.shape
            seq_len = h * w

            expected_in_ch = self.in_chs_for_proj[i]
            assert Cc == expected_in_ch, f"scale {i} expect in-ch {expected_in_ch}, got {Cc}"

            # x_seq = cur.view(BV2, Cc, seq_len).permute(0, 2, 1).contiguous()  # (BV, seq_len, C_in)

            # project to d_model
            # x_proj = self.proj_in(x_seq)  # (BV, seq_len, d_model)
            x_proj = self.proj_ins[i](cur).view(BV2, self.d_model, seq_len).permute(0, 2, 1).contiguous()  # (BV2, seq_len, d_model)

            # add pos emb if present
            if self.use_pos_emb:
                pos = self.pos_embs[i]
                if pos.shape[1] == seq_len:
                    x_proj = x_proj + pos.to(dtype=x_proj.dtype, device=device)
                else:
                    # fallback: zeros (safe)
                    x_proj = x_proj

            # transformer encode (per-view attention because batch=B*V)
            x_enc = self.scale_encoders[i](x_proj)  # (BV, seq_len, d_model)

            # project back to out channels of this scale
            x_out = self.proj_outs[i](x_enc)  # (BV, seq_len, out_ch_i)
            feat_map = x_out.permute(0, 2, 1).contiguous().view(BV2, self.out_channels_per_scale[i], h, w)
            feats.append(feat_map)

            # prepare cur for next scale (avg pool if divisible else interpolate)
            if i < len(self.scales) - 1:
                next_h, next_w = self.scales[i + 1]
                factor_h = h // next_h if next_h > 0 else None
                factor_w = w // next_w if next_w > 0 else None
                if (factor_h is not None and factor_w is not None and
                    factor_h >= 1 and factor_w >= 1 and (h % next_h == 0) and (w % next_w == 0)):
                    if factor_h == 1 and factor_w == 1:
                        cur = feat_map
                    else:
                        cur = F.avg_pool2d(feat_map, kernel_size=(factor_h, factor_w), stride=(factor_h, factor_w))
                else:
                    cur = F.interpolate(feat_map, size=(next_h, next_w), mode='bilinear', align_corners=False)

        # feats: list of (B*V, out_ch_i, H_s, W_s)
        return feats


# -----------------------------
# Cross-view aggregator
# -----------------------------
class CrossViewAggregator(nn.Module):
    """
    Cross-view Fusion Transformer aggregator.

    Inputs:
      - tokens: last_hidden_state from backbone, shape (B', seq_len, embed_dim)
      - is_multiview: whether original input was multiview
      - B: original batch size when multiview (required if is_multiview True)
      - n_views: number of views per sample (required if is_multiview True)

    Returns:
      - feat_map: (B*n_views, out_channels, upsample_size, upsample_size)
    """
    def __init__(
        self,
        embed_dim: int,
        num_patches_per_side: int,
        out_channels: int = None,
        hierarchical_scales: list = [(48,48),(24,24),(12,12)],
        upsample_size: int = 96,
        # transformer params
        d_model: int = 256,
        per_view_layers: int = 2,
        view_fusion_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        use_pos_emb: bool = True,
        hierarchical=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches_per_side = num_patches_per_side
        self.seq_len = num_patches_per_side * num_patches_per_side
        self.upsample_size = upsample_size

        # set output channels: if None, keep embed_dim to preserve behavior
        self.out_channels = out_channels if out_channels is not None else embed_dim

        # down / up projections
        self.down_proj = nn.Linear(self.embed_dim, d_model)
        self.up_proj = nn.Linear(d_model, self.out_channels)

        # optional learned 2D positional embedding (shared across views)
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
            nn.init.normal_(self.pos_emb, std=0.02)

        # per-view transformer encoder (applied to each view separately)
        if per_view_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                # norm_first is okay if PyTorch supports; fallback to default if not
            )
            self.per_view_encoder = nn.TransformerEncoder(layer, num_layers=per_view_layers)
        else:
            self.per_view_encoder = None

        # view-fusion transformer (applied per patch across views)
        if view_fusion_layers > 0:
            vf_heads = max(1, min(nhead, d_model // 16))
            vlayer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=vf_heads,
                dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.view_fusion_encoder = nn.TransformerEncoder(vlayer, num_layers=view_fusion_layers)
        else:
            self.view_fusion_encoder = None

        if hierarchical:
            if hierarchical_scales is None:
                hierarchical_scales = [
                    (self.upsample_size, self.upsample_size),
                    (self.upsample_size // 2, self.upsample_size // 2),
                    (self.upsample_size // 4, self.upsample_size // 4),
                    (self.upsample_size // 8, self.upsample_size // 8),
                ]
            self.hierarchical_agg = HierarchicalTransformerAggregation(
                in_ch=self.out_channels,                      # 256
                out_channels_per_scale=[512, 1024, 2048],
                d_model=512,                                  # 512 % nhead should == 0
                nhead=4,
                layers_per_scale=1,                           # 每尺度 transformer 层数（可调）
                scales=[(48,48),(24,24),(12,12)],
                use_pos_emb=True,
            )
        else:
            self.hierarchical_agg = None

        # small conv to refine after upsample
        self.post_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # store d_model for reshapes
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor, is_multiview: bool = False, B: int = None, n_views: int = 1):
        """
        tokens: (B', seq_len, embed_dim)
        If is_multiview True, B and n_views must be provided such that B'* = B * n_views
        """
        Bprime, seq_len, embed_dim = tokens.shape
        assert seq_len == self.seq_len, f"seq_len mismatch: {seq_len} vs expected {self.seq_len}"
        if is_multiview:
            assert (B is not None) and (n_views is not None), "B and n_views required when is_multiview=True"
            assert Bprime == B * n_views, f"token batch mismatch: {Bprime} != B*n_views ({B}*{n_views})"
        else:
            # treat as B x 1 views
            if B is None:
                B = Bprime
            n_views = 1

        device = tokens.device
        # reshape to (B, n_views, seq_len, embed_dim)
        tokens_view = tokens.view(B, n_views, seq_len, embed_dim)

        # flatten per-view for down-proj and per-view encoder: (B*n_views, seq_len, embed_dim)
        B_nv = B * n_views
        tokens_flat = tokens_view.view(B_nv, seq_len, embed_dim)

        # down project to d_model
        x = self.down_proj(tokens_flat)  # (B*nv, seq_len, d_model)

        # add position embedding shared across views
        if self.use_pos_emb:
            x = x + self.pos_emb.to(dtype=x.dtype, device=device)

        # per-view encoding
        if self.per_view_encoder is not None:
            x = self.per_view_encoder(x)  # (B*nv, seq_len, d_model)

        # reshape to (B, n_views, seq_len, d_model)
        x_views = x.view(B, n_views, seq_len, self.d_model)

        # prepare per-patch cross-view fusion:
        # (B, seq_len, n_views, d_model) -> (B*seq_len, n_views, d_model)
        x_perm = x_views.permute(0, 2, 1, 3).contiguous()
        x_pf = x_perm.view(B * seq_len, n_views, self.d_model)

        if self.view_fusion_encoder is not None:
            x_pf = self.view_fusion_encoder(x_pf)  # (B*seq_len, n_views, d_model)

        # restore to (B, n_views, seq_len, d_model)
        x_pf = x_pf.view(B, seq_len, n_views, self.d_model).permute(0, 2, 1, 3).contiguous()
        x_final = x_pf.view(B_nv, seq_len, self.d_model)  # (B*n_views, seq_len, d_model)

        # up project to out channels
        up = self.up_proj(x_final)  # (B*n_views, seq_len, out_ch)

        # unpatchify: (B*n_views, out_ch, Hp, Wp)
        Hp = Wp = self.num_patches_per_side
        feat_map = up.transpose(1, 2).reshape(B_nv, self.out_channels, Hp, Wp)

        # upsample to desired size and post conv
        feat_map_agg = F.interpolate(feat_map, size=(self.upsample_size, self.upsample_size), mode='bilinear', align_corners=False)
        feat_map = self.post_conv(feat_map_agg)

        if self.hierarchical_agg is not None:
            feat_map = self.hierarchical_agg(feat_map)  # list of (B*V, C, Hi, Wi)
            feat_map = [feat_map_agg] + feat_map
          
        return feat_map


class LevelAdapterModule(nn.Module):
    def __init__(self, copy_layer, in_dim, out_dim, adapter_depth=2, adapter_nhead=8,
    adapter_ffn_dim=None, dropout=0.1):
        super().__init__()
        self.copy_head = copy_layer
        self.in_dim = in_dim
        self.out_dim = out_dim

        # projection: norm -> linear -> norm
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        # adapter stack: ModuleList of TransformerEncoderLayer
        if adapter_ffn_dim is None:
            adapter_ffn_dim = max(out_dim * 4, 2048)
        layers = []
        for _ in range(max(0, adapter_depth)):
            layers.append(nn.TransformerEncoderLayer(d_model=out_dim,
                                                    nhead=adapter_nhead,
                                                    dim_feedforward=adapter_ffn_dim,
                                                    dropout=dropout,
                                                    activation='gelu',
                                                    batch_first=False))  # note: we'll feed seq-first
        # store as ModuleList for explicit per-layer application
        self.adapter_layers = nn.ModuleList(layers)

        # final normalization (optional)
        self.final_norm = nn.LayerNorm(out_dim)

    def forward(self, token_src):
        """
        token_src: (B', seq_len, in_dim)
        returns: adapted (B', seq_len, out_dim)
        """
        # apply copy head
        # different encoder layer implementations may return tuple or tensor
        try:
            copy_out = self.copy_head(token_src, attention_mask=None, output_attentions=False)
            # many implementations return tuple-like where first item is hidden_states
            if isinstance(copy_out, (tuple, list)):
                copy_out = copy_out[0]
        except TypeError:
            # fallback: try call with single arg or direct call
            try:
                out_tmp = self.copy_head(token_src)
                copy_out = out_tmp[0] if isinstance(out_tmp, (tuple, list)) else out_tmp
            except Exception:
                # as last resort assume copy_head is identity-like
                copy_out = token_src

        # projection
        proj_out = self.proj(copy_out)  # (B', seq_len, out_dim)

        # adapter: nn.TransformerEncoderLayer expects (seq_len, batch, embed) when batch_first=False
        # Using batch_first=False in construction: pass (seq_len, batch, embed)
        if len(self.adapter_layers) == 0:
            adapted = proj_out
        else:
            # transpose to (seq_len, B', out_dim)
            adapted_t = proj_out.transpose(0, 1)
            for layer in self.adapter_layers:
                adapted_t = layer(adapted_t)
            adapted = adapted_t.transpose(0, 1)  # back to (B', seq_len, out_dim)

        adapted = self.final_norm(adapted)
        return adapted



@MODELS.register_module()
class SigLipBackboneAdapter(nn.Module):
    """
    Adapter: wrap SigLip vision model and return 2D feature map(s) compatible with
    architectures expecting (B, C, Hf, Wf) or list of such tensors.
    Params:
      pretrained_name: name or path for SigLip pretrained weights
      out_channels: if not None, project SigLip embed_dim -> out_channels via 1x1 conv
      return_list: if True, return [feat_map] so it matches neck expecting a list
      freeze: if True, freeze SigLip weights (optional)
      use_tower: if True, instantiate SigLipVisionTower variant (if available)
    """
    def __init__(self, base_model_name, copy_indices, out_channels,
            upsample_size = [(96, 96), (48,48),(24,24),(12,12)],
            adapter_depths=2, adapter_nhead=8, adapter_ffn_dim=None,
            freeze_backbone=True, dropout=0.1):
        super().__init__()

        assert len(upsample_size) == len(out_channels)

        vision_cfg = SigLipVisionConfig()
        self.model = SigLipVisionTower(base_model_name, vision_cfg)
        self._is_tower = True
    
        self.copy_indices = list(copy_indices)
        self.out_channels = list(out_channels)
        self.upsample_size = list(upsample_size)

        # get config/embed_dim/encoder layers safely
        model_cfg = getattr(self.model, "config", None)
        if model_cfg is None:
            model_cfg = getattr(getattr(self.model, "vision_tower", self.model), "config", None)
        if model_cfg is None:
            raise RuntimeError("无法从 model 获取 config")
        self.embed_dim = getattr(model_cfg, "hidden_size", None)
        self.image_size = getattr(model_cfg, "image_size", None) 
        self.patch_size = getattr(model_cfg, "patch_size", None) 
        self.num_patches_per_side = self.image_size // self.patch_size

        if self.embed_dim is None:
            raise RuntimeError("无法从 config 获取 hidden_size")

        # get encoder layers list if available
        vision_base = None
        if hasattr(self.model, "vision_tower") and hasattr(self.model.vision_tower, "vision_model"):
            vision_base = self.model.vision_tower.vision_model
        # elif hasattr(self.model, "vision_model"):
        #     vision_base = self.model.vision_model
        # elif hasattr(self.model, "encoder"):
        #     vision_base = self.model
        else:
            vision_base = None

        encoder_layers = None
        if vision_base is not None and hasattr(vision_base, "encoder") and hasattr(vision_base.encoder, "layers"):
            encoder_layers = vision_base.encoder.layers

        if encoder_layers is None:
            raise RuntimeError("找不到 vision encoder layers，无法构造 copy_heads。请检查 base_model 的结构。")

        # build level modules
        self.levels = nn.ModuleList()
        for i, idx in enumerate(self.copy_indices):
            # fetch source layer; if idx out of current range, try model.removed_encoder_layer
            if 0 <= idx < len(encoder_layers):
                layer_src = encoder_layers[idx]
            else:
                raise IndexError(f"copy index {idx} 越界，encoder layers 长度 {len(encoder_layers)}，且没有 removed_encoder_layer")
            
            layer_copy = copy.deepcopy(layer_src)
            depth = adapter_depths[i] if isinstance(adapter_depths, (list, tuple)) else adapter_depths
            lvl = LevelAdapterModule(copy_layer=layer_copy,
                                    in_dim=self.embed_dim,
                                    out_dim=self.out_channels[i],
                                    adapter_depth=depth,
                                    adapter_nhead=adapter_nhead,
                                    adapter_ffn_dim=adapter_ffn_dim,
                                    dropout=dropout) 
            self.levels.append(lvl)

        # freeze base model parameters if requested
        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

        # device helper
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

    def _call_model_get_states(self, x):

        outs = self.model.vision_tower(x, output_hidden_states=True, return_dict=True)

        hidden_states = getattr(outs, "hidden_states", None)
        last_hidden = getattr(outs, "last_hidden_state", None)

        return hidden_states, last_hidden

    def forward(self, x):
        """
        x: (B, C, H, W) 或 (B, n_views, C, H, W)
        返回: list of adapted tokens:
            each element shape = (B, n_views, seq_len, out_ch)  (if n_views>1)
                            or (B, seq_len, out_ch)           (if n_views==1)
        """
        is_multiview = (x.dim() == 5)
        if is_multiview:
            B, n_views, C, H, W = x.shape
            x_in = x.reshape(B * n_views, C, H, W)
        elif x.dim() == 4:
            B, C, H, W = x.shape
            n_views = 1
            x_in = x
        else:
            raise ValueError("Unsupported input dimensions")

        device = next(self.model.parameters()).device
        x_in = x_in.to(device=device)

        hidden_states, last_hidden = self._call_model_get_states(x_in)

        if hidden_states is None and last_hidden is None:
            raise RuntimeError("无法从底层模型获取 hidden states 或 last_hidden_state")

        results = []
        for k, idx in enumerate(self.copy_indices):
            # choose token source: prefer hidden_states[idx+1] (embeddings at 0), 否则 fallback last_hidden
            if hidden_states is not None:
                # hidden_states typically length = num_layers + 1 (0: embeddings)
                target_pos = idx  if idx < len(hidden_states) else -1
                token_src = hidden_states[target_pos]

            # ensure token_src shape (B'* , seq_len, embed_dim)
            # apply corresponding level module
            lvl = self.levels[k]
            adapted = lvl(token_src)  # (B'* , seq_len, out_ch)

            # reshape back to include view dimension if needed
            if is_multiview:
                Hp = Wp = self.num_patches_per_side
                adapted = adapted.transpose(1, 2).reshape(B * n_views, self.out_channels[k], Hp, Wp)       
                adapted = F.interpolate(adapted, size=self.upsample_size[k], mode='bilinear', align_corners=False)
            else:
                adapted = adapted.view(B, adapted.size(1), adapted.size(2))
            results.append(adapted)

        last_hidden_adapted = last_hidden.transpose(1, 2).reshape(B * n_views, self.out_channels[-1], Hp, Wp)
        last_hidden_adapted = F.interpolate(last_hidden_adapted, size=self.upsample_size[-1], mode='bilinear', align_corners=False)
        results.append(last_hidden_adapted)

        return results, last_hidden.transpose(1, 2).reshape(B * n_views, self.out_channels[-1], Hp, Wp)


if __name__ == "__main__":
    # Debug entry: construct adapter and run with input shape (1, 32, 3, 384, 384)
    import torch

    # Create dummy multiview input: batch=1, n_views=32, C=3, H=384, W=384
    dummy_input = torch.randn(1, 32, 3, 384, 384)

    # Instantiate adapter:
    # - pretrained_name=None -> use local random-initialized SigLip model (fast, no download)
    # - out_channels=None -> keep original embed_dim (1152)
    # - return_list=False -> return tensor (B*n_views, C, Hp, Wp)
    adapter = SigLipBackboneAdapter(pretrained_name="google/siglip-so400m-patch14-384", out_channels=None, return_list=False, freeze=True, use_tower=True)

    # Run forward
    out = adapter(dummy_input)

    # Print shapes for debugging
    print("Input shape:", dummy_input.shape)
    print("Output dtype/device:", out.dtype, out.device)
    print("Output shape:", out.shape)
    # If multiview, the adapter returns merged batch shape B*n_views as first dim
    # Compute expected Hp/Wp and channels
    cfg = adapter.model.config if hasattr(adapter.model, "config") else adapter.model.vision_model.config
    Hp = Wp = cfg.image_size // cfg.patch_size
    print("Expected patch grid:", Hp, "x", Wp)
    expected_seq = Hp * Wp
    print("Expected seq len:", expected_seq)
