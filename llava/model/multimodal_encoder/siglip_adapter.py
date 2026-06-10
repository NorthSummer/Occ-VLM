import torch
from torch import nn
import torch.nn.functional as F
import copy

from .siglip_encoder import SigLipVisionTower, SigLipVisionModel, SigLipVisionConfig
from mmengine.model import BaseModel
from llava.model.multimodal_encoder.embodiedscan.registry import MODELS


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

        return results


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
