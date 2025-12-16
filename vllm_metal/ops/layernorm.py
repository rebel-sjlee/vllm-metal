# SPDX-License-Identifier: Apache-2.0
"""Layer normalization operations for Metal backend."""

import torch


def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> None:
    """Root Mean Square Layer Normalization.

    Computes: out = (input / sqrt(mean(input^2) + epsilon)) * weight

    This is the normalization used in LLaMA, Mistral, and other models.

    Args:
        out: Output tensor
        input: Input tensor
        weight: Scale weight tensor
        epsilon: Small constant for numerical stability
    """
    # Compute RMS
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    input_normalized = input * torch.rsqrt(variance + epsilon)

    # Apply weight
    out.copy_(input_normalized * weight)


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> None:
    """Fused residual addition and RMS normalization.

    Computes:
        input = input + residual
        input = rms_norm(input, weight, epsilon)

    This modifies input in-place for both the residual addition
    and the normalization.

    Args:
        input: Input tensor (modified in-place)
        residual: Residual tensor to add
        weight: Scale weight tensor
        epsilon: Small constant for numerical stability
    """
    # Add residual in-place
    input.add_(residual)

    # Compute RMS norm in-place
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    input.mul_(torch.rsqrt(variance + epsilon))
    input.mul_(weight)


def layer_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float = 1e-5,
) -> None:
    """Standard Layer Normalization.

    Computes: out = (input - mean) / sqrt(var + epsilon) * weight + bias

    Args:
        out: Output tensor
        input: Input tensor
        weight: Scale weight tensor
        bias: Bias tensor
        epsilon: Small constant for numerical stability
    """
    normalized_shape = input.shape[-1:]
    result = torch.nn.functional.layer_norm(
        input, normalized_shape, weight, bias, epsilon
    )
    out.copy_(result)


def fused_add_layer_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float = 1e-5,
) -> None:
    """Fused residual addition and layer normalization.

    Args:
        input: Input tensor (modified in-place)
        residual: Residual tensor to add
        weight: Scale weight tensor
        bias: Bias tensor
        epsilon: Small constant for numerical stability
    """
    # Add residual in-place
    input.add_(residual)

    # Apply layer norm
    normalized_shape = input.shape[-1:]
    result = torch.nn.functional.layer_norm(
        input, normalized_shape, weight, bias, epsilon
    )
    input.copy_(result)
