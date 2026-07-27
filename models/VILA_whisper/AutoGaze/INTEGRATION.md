# Integrating AutoGaze into Vision Transformers

This guide explains how to modify an existing image ViT to work with AutoGaze. We use [our SigLIP implementation](autogaze/vision_encoders/siglip/) as a running example, comparing it against [the original SigLIP from HuggingFace Transformers (v4.51.1)](https://github.com/huggingface/transformers/blob/v4.51.1/src/transformers/models/siglip/modeling_siglip.py).

## Overview

AutoGaze predicts which patches to attend to (the "gaze") for each video frame. The ViT then only processes the gazed patches of all frames rather than all patches of a single image as usual. This requires two changes to a standard image ViT:

1. **Patch Embedding**: Instead of embedding all patches, only embed the gazed patches selected by AutoGaze for each frame.
2. **Attention Mask**: Since we are retargeting an image ViT into a video ViT, we need to construct an attention mask to control how tokens from different frames interact. Three types are supported: block-causal (default), causal, and bidirectional.

The rest of the ViT (encoder layers, attention, MLP, layer norms, etc.) stays **unchanged**.

An example of using SigLIP+AutoGaze can be found in `QUICK_START.md`.

## `gazing_info`

AutoGaze outputs a `gazing_info` dict that is passed to the ViT. It contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `gazing_pos` | `(B, N)` | Indices of the gazed patches across all frames, where `N = sum(num_gazing_each_frame)`. |
| `num_gazing_each_frame` | `(T,)` | Number of gazed patches per frame (including padding). |
| `if_padded_gazing` | `(B, N)` | Boolean mask indicating which positions are padding (not real gazes). |

## Step 1: Modify Patch Embedding

In a standard ViT, patch embedding converts *all* image patches into tokens. With AutoGaze, we only keep the patches at positions specified by `gazing_info['gazing_pos']`.

### What to change

Add a method to **select only the gazed patches** from the full sequence of patches (and their corresponding position embeddings). In our SigLIP implementation, this is `mask_with_gazing`:

```python
def mask_with_gazing(self, sequence, gazing_info):
    """
    Select only the gazed patches from the full sequence.
    Padded gazing positions are mapped to a dummy token (index 0).
    """
    gazing_pos = gazing_info['gazing_pos'].clone()
    if_padded_gazing = gazing_info['if_padded_gazing'].clone()

    B = sequence.shape[0]

    # Map padded gazing positions to a dummy token
    gazing_pos[if_padded_gazing] = 0

    # Gather only the gazed tokens
    sequence_gazed = sequence[torch.arange(B)[:, None], gazing_pos]
    return sequence_gazed
```

Then in the embedding forward pass, after computing all patches and position embeddings, apply this selection:

```python
# Compute all patches and position embeddings (across frames and scales)
# patches: (B, T*num_patches, patch_dim)
# pos_embeddings: (B, T*num_patches, embed_dim)

# Select only the gazed patches
patches = self.mask_with_gazing(patches, gazing_info)
pos_embeddings = self.mask_with_gazing(pos_embeddings, gazing_info)

# Then embed and add position embeddings as usual
embeddings = linear(patches) + pos_embeddings
```

**Key point**: The input `pixel_values` shape changes from `(B, C, H, W)` to `(B, T, C, H, W)` since AutoGaze operates on video frames. Patches from all frames are flattened into a single sequence before gazing selection.

> See `SiglipVisionEmbeddings` in [`modeling_siglip.py`](autogaze/vision_encoders/siglip/modeling_siglip.py) for the complete implementation.

### Multi-Scale Patchification

AutoGaze supports multi-scale patches (e.g., `32+64+112+224`), where the video is resized to each scale and patchified independently. The patches and position embeddings from all scales are concatenated before gazing selection. This is handled in `get_gazed_patches_and_other_embeddings` in our SigLIP. If you only use a single scale, this simply reduces to the standard single-resolution patchification.

## Step 2: Construct the Attention Mask

Since we are repurposing an image ViT to process multiple video frames as a single sequence, we need an attention mask to control how tokens from different frames interact. We support three attention types (configured via `attn_type`):

| `attn_type` | Inter-frame attention | Intra-frame attention | Description |
|---|---|---|---|
| `block_causal` | Causal (past frames only) | Bidirectional | Tokens attend to all tokens in the same frame and all tokens from previous frames. **Recommended default.** |
| `causal` | Causal | Causal | Strictly causal — each token attends only to preceding tokens in the flattened sequence. |
| `bidirectional` | Bidirectional | Bidirectional | Full attention across all tokens (all frames see each other). |

In all cases, padded gazing tokens are masked out so they are not attended to.

### Attention backend compatibility

Not all attention types work with all backends:

| Backend | Supported `attn_type` |
|---|---|
| `flash_attention_2` | `causal`, `bidirectional` |
| `sdpa`, `eager`, `flex_attention` | `block_causal` |

The reason is that `flash_attention_2` natively supports causal masking (via the `is_causal` flag) and simple padding masks, but does not accept arbitrary 2D attention masks needed for block-causal attention. Conversely, the other backends construct an explicit `(B, num_heads, N, N)` additive mask, which can express block-causal patterns but would be redundant for the simpler causal/bidirectional cases that flash attention handles more efficiently.

### What to change

Add a method to construct the appropriate attention mask. In our SigLIP, this is `get_causal_mask` in `SiglipVisionTransformer`. Here is a simplified version showing the block-causal case:

```python
def get_causal_mask(self, num_tokens_each_frame, batch_size, num_heads,
                    token_mask=None, dtype=torch.float32):
    T = len(num_tokens_each_frame)
    N = num_tokens_each_frame.sum()

    # Start with a causal (lower-triangular) mask
    mask = torch.tril(torch.ones(batch_size, N, N, dtype=dtype))

    # Allow full bidirectional attention within each frame
    for t in range(T):
        start = num_tokens_each_frame[:t].sum()
        end = num_tokens_each_frame[:t+1].sum()
        mask[:, start:end, start:end] = 1

    # Zero out columns for padded tokens
    if token_mask is not None:
        mask = mask * (~token_mask.unsqueeze(1)).to(dtype)

    # Convert to additive mask (0 for attend, -inf for ignore)
    mask = torch.where(mask == 1, 0, -torch.inf).to(dtype)
    mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
    return mask
```

Then in the transformer forward pass, construct and pass this mask to the encoder:

```python
encoder_attn_mask = self.get_causal_mask(
    gazing_info['num_gazing_each_frame'],
    batch_size=B,
    num_heads=self.config.num_attention_heads,
    token_mask=gazing_info['if_padded_gazing'],
    dtype=pixel_values.dtype,
)
encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=encoder_attn_mask)
```

> See `SiglipVisionTransformer.get_causal_mask` in [`modeling_siglip.py`](autogaze/vision_encoders/siglip/modeling_siglip.py) for the complete implementation covering all three attention types and backends.

## Step 3: Update Config and Forward Signatures

### Configuration

Add these fields to your vision config (see [`configuration_siglip.py`](autogaze/vision_encoders/siglip/configuration_siglip.py)):

- `scales` (str): Multi-scale resolutions separated by `+`, e.g., `'32+64+112+224'`. Use `'224'` for single-scale.
- `attn_type` (str): Attention type — `'block_causal'` (default; causal across frames, bidirectional within each frame), `'causal'` (strictly causal), or `'bidirectional'` (full attention). See Step 2 for details and backend compatibility.
- `frame_independent_encoding` (bool): If `True`, tokens from different frames cannot attend to each other (only intra-frame attention).

### Forward signature

Add `gazing_info: Optional[dict] = None` to the forward methods of both the embedding module and the transformer module.

## Summary of Changes

| Component | Original ViT | With AutoGaze |
|-----------|-------------|---------------|
| **Input shape** | `(B, C, H, W)` | `(B, T, C, H, W)` |
| **Patch embedding** | Embeds all patches | Embeds only gazed patches via `mask_with_gazing` |
| **Attention mask** | None (full attention) | Block-causal / causal / bidirectional mask from `get_causal_mask` |
| **Encoder / MLP / LayerNorm** | No change | No change |
| **Config** | Standard | + `scales`, `attn_type`, `frame_independent_encoding` |

<br>
<br>
<br>

# Integrating AutoGaze into MLLMs

After adding AutoGaze to a ViT, it's conceptually trivial to use it in an MLLM--all you need is to send the ViT features of gazed patches into the LLM.

As an example, we open-sourced NVILA-HD-Video, an MLLM with AutoGaze. See [instructions](https://github.com/NVlabs/VILA/tree/main/vila_hd/nvila_hd_video) in the VILA repo for using NVILA-HD-Video.

