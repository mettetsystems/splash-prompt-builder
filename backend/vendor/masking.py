"""Eager-only bidirectional masking for checkpoints expecting the Transformers 5 API.

The 4D mask is already prepared by LLaDA's sampler and must pass through unchanged.
The 2D path expands padding; it deliberately introduces no causal triangle.
"""
def create_bidirectional_mask(config, inputs_embeds, attention_mask=None, **kwargs):
    if config._attn_implementation != "eager":
        raise ValueError("Splash's compatibility mask supports eager attention only.")
    if attention_mask is None or attention_mask.ndim == 4:
        return attention_mask
    if attention_mask.ndim != 2:
        raise ValueError("Expected a 2D padding mask or prepared 4D mask.")
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
    return _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype, tgt_len=inputs_embeds.shape[1])
