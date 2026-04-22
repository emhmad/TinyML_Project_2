"""
Reusable ViT attention-rollout extractor.

Originally this logic lived inside `experiments/e_attention_viz.py` and
was only used to draw three cherry-picked melanoma overlays. W12 asks
us to compute attention-lesion overlap (IoU + pointing game) across the
full melanoma validation set, which means the rollout extractor has to
be importable as a library, safe to reuse across many samples, and
cleanly re-entrant. This module is that library.

The rollout implementation is the standard Abnar-Zuidema construction:
  - forward hook on each block's attention dropout captures the post-
    softmax attention map,
  - average attention heads,
  - add the identity (residual path),
  - renormalise,
  - chain-multiply block-to-block to obtain token-to-token flow,
  - read off the CLS-token row and reshape back to the patch grid.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class AttentionRollout:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.attentions: list[torch.Tensor] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self._orig_fused_flags: dict[int, bool] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        for block in self.model.blocks:
            attn = block.attn
            if hasattr(attn, "fused_attn"):
                self._orig_fused_flags[id(attn)] = bool(attn.fused_attn)
                attn.fused_attn = False
            if hasattr(attn, "attn_drop"):
                self.handles.append(attn.attn_drop.register_forward_hook(self._hook_fn))
            else:
                raise RuntimeError("Attention module does not expose attn_drop; rollout hook cannot be attached.")

    def _hook_fn(self, module, inputs, output) -> None:
        self.attentions.append(output.detach().cpu())

    def clear(self) -> None:
        self.attentions.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for block in self.model.blocks:
            attn = block.attn
            if hasattr(attn, "fused_attn") and id(attn) in self._orig_fused_flags:
                attn.fused_attn = self._orig_fused_flags[id(attn)]

    def get_rollout(self, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
        """
        Run the model and return a HxW heatmap in [0, 1] upsampled to the
        input image resolution.
        """
        self.clear()
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image_tensor.to(device, non_blocking=True))
        if not self.attentions:
            raise RuntimeError("No attention tensors were collected during rollout.")

        joint_attention = None
        for attention in self.attentions:
            attn = attention.mean(dim=1)[0]
            identity = torch.eye(attn.size(-1), dtype=attn.dtype)
            attn = 0.5 * attn + 0.5 * identity
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            joint_attention = attn if joint_attention is None else attn @ joint_attention

        prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        spatial_tokens = joint_attention[0, prefix_tokens:]
        grid_h, grid_w = self.model.patch_embed.grid_size
        rollout = spatial_tokens.reshape(grid_h, grid_w)
        rollout = F.interpolate(
            rollout.unsqueeze(0).unsqueeze(0),
            size=(int(image_tensor.shape[-2]), int(image_tensor.shape[-1])),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        rollout = rollout - rollout.min()
        rollout = rollout / rollout.max().clamp_min(1e-8)
        return rollout.numpy()
