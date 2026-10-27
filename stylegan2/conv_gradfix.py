# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""NVIDIA gradfix adapted to modern ATen, enabled only for PL/R1.

Derived from vendor/torch_utils/ops/conv2d_gradfix.py at the pinned NVIDIA
commit (see SOURCES.md). Replaces removed cuDNN weight-gradient entry points
with built-in ATen convolution_backward. No CUDA extension or extra package.
Native forward and ordinary training phases remain unchanged. Equivalent
real-arithmetic derivatives do not imply bitwise identity of GPU kernels.
"""

from torch_utils.ops import conv2d_gradfix as legacy
import contextlib
import torch

# pylint: disable=redefined-builtin
# pylint: disable=arguments-differ
# pylint: disable=protected-access

#----------------------------------------------------------------------------

enabled = False                     # Enable the custom op by setting this to true.
#----------------------------------------------------------------------------

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if _should_use_custom_op(input):
        return _conv2d_gradfix(transpose=False, weight_shape=weight.shape, stride=stride, padding=padding, output_padding=0, dilation=dilation, groups=groups).apply(input, weight, bias)
    return torch.nn.functional.conv2d(input=input, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    if _should_use_custom_op(input):
        return _conv2d_gradfix(transpose=True, weight_shape=weight.shape, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation).apply(input, weight, bias)
    return torch.nn.functional.conv_transpose2d(input=input, weight=weight, bias=bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)

#----------------------------------------------------------------------------

def _should_use_custom_op(input):
    return enabled


def _tuple_of_ints(xs, ndim):
    xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,) * ndim
    assert len(xs) == ndim
    assert all(isinstance(x, int) for x in xs)
    return xs

#----------------------------------------------------------------------------

_conv2d_gradfix_cache = dict()

def _conv2d_gradfix(transpose, weight_shape, stride, padding, output_padding, dilation, groups):
    # Parse arguments.
    ndim = 2
    weight_shape = tuple(weight_shape)
    stride = _tuple_of_ints(stride, ndim)
    padding = _tuple_of_ints(padding, ndim)
    output_padding = _tuple_of_ints(output_padding, ndim)
    dilation = _tuple_of_ints(dilation, ndim)

    # Lookup from cache.
    key = (transpose, weight_shape, stride, padding, output_padding, dilation, groups)
    if key in _conv2d_gradfix_cache:
        return _conv2d_gradfix_cache[key]

    # Validate arguments.
    assert groups >= 1
    assert len(weight_shape) == ndim + 2
    assert all(stride[i] >= 1 for i in range(ndim))
    assert all(padding[i] >= 0 for i in range(ndim))
    assert all(dilation[i] >= 0 for i in range(ndim))
    if not transpose:
        assert all(output_padding[i] == 0 for i in range(ndim))
    else: # transpose
        assert all(0 <= output_padding[i] < max(stride[i], dilation[i]) for i in range(ndim))

    # Helpers.
    common_kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    def calc_output_padding(input_shape, output_shape):
        if transpose:
            return [0, 0]
        return [
            input_shape[i + 2]
            - (output_shape[i + 2] - 1) * stride[i]
            - (1 - 2 * padding[i])
            - dilation[i] * (weight_shape[i + 2] - 1)
            for i in range(ndim)
        ]

    # Forward & backward.
    class Conv2d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            assert weight.shape == weight_shape
            if not transpose:
                output = torch.nn.functional.conv2d(input=input, weight=weight, bias=bias, **common_kwargs)
            else: # transpose
                output = torch.nn.functional.conv_transpose2d(input=input, weight=weight, bias=bias, output_padding=output_padding, **common_kwargs)
            ctx.save_for_backward(input, weight)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            grad_input = None
            grad_weight = None
            grad_bias = None

            if ctx.needs_input_grad[0]:
                p = calc_output_padding(input_shape=input.shape, output_shape=grad_output.shape)
                grad_input = _conv2d_gradfix(transpose=(not transpose), weight_shape=weight_shape, output_padding=p, **common_kwargs).apply(grad_output, weight, None)
                assert grad_input.shape == input.shape

            if ctx.needs_input_grad[1] and not legacy.weight_gradients_disabled:
                grad_weight = Conv2dGradWeight.apply(grad_output, input)
                assert grad_weight.shape == weight_shape

            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum([0, 2, 3])

            return grad_input, grad_weight, grad_bias

    # Gradient with respect to the weights.
    class Conv2dGradWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx, grad_output, input):
            # Modern ATen equivalent of NVIDIA's removed cuDNN weight-gradient op.
            # Only weight gradient is requested; values of the placeholder weight
            # are not read by this derivative (only its shape/dtype/device).
            placeholder = input.new_empty(weight_shape)
            grad_weight = torch.ops.aten.convolution_backward.default(
                grad_output, input, placeholder, None, stride, padding, dilation,
                transpose, output_padding, groups, [False, True, False])[1]
            assert grad_weight.shape == weight_shape
            ctx.save_for_backward(grad_output, input)
            return grad_weight

        @staticmethod
        def backward(ctx, grad2_grad_weight):
            grad_output, input = ctx.saved_tensors
            grad2_grad_output = None
            grad2_input = None

            if ctx.needs_input_grad[0]:
                grad2_grad_output = Conv2d.apply(input, grad2_grad_weight, None)
                assert grad2_grad_output.shape == grad_output.shape

            if ctx.needs_input_grad[1]:
                p = calc_output_padding(input_shape=input.shape, output_shape=grad_output.shape)
                grad2_input = _conv2d_gradfix(transpose=(not transpose), weight_shape=weight_shape, output_padding=p, **common_kwargs).apply(grad_output, grad2_grad_weight, None)
                assert grad2_input.shape == input.shape

            return grad2_grad_output, grad2_input

    _conv2d_gradfix_cache[key] = Conv2d
    return Conv2d

#----------------------------------------------------------------------------


@contextlib.contextmanager
def regularization(enabled_for_phase=True):
    """Use only during PL/R1 forward+backward; ordinary phases stay native."""
    global enabled
    previous = enabled
    enabled = enabled_for_phase
    try:
        yield
    finally:
        enabled = previous


def install():
    legacy.conv2d = conv2d
    legacy.conv_transpose2d = conv_transpose2d
