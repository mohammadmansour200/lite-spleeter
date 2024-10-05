#!/usr/bin/env python
# coding: utf8

"""Utility function for tensorflow."""

from typing import Any


# pyright: reportMissingImports=false
# pylint: disable=import-error
import tensorflow as tf  # type: ignore

# pylint: enable=import-error

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


def from_float32_to_uint8(
    tensor: tf.Tensor,
    tensor_key: str = "tensor",
    min_key: str = "min",
    max_key: str = "max",
) -> tf.Tensor:
    tensor_min = tf.reduce_min(tensor)
    tensor_max = tf.reduce_max(tensor)
    return {
        tensor_key: tf.cast(
            (tensor - tensor_min) / (tensor_max - tensor_min + 1e-16) * 255.9999,
            dtype=tf.uint8,
        ),
        min_key: tensor_min,
        max_key: tensor_max,
    }


def from_uint8_to_float32(
    tensor: tf.Tensor, tensor_min: tf.Tensor, tensor_max: tf.Tensor
) -> tf.Tensor:
    return (
        tf.cast(tensor, tf.float32) * (tensor_max - tensor_min) / 255.9999 + tensor_min
    )


def pad_and_partition(tensor: tf.Tensor, segment_len: int) -> tf.Tensor:
    """
    Pad and partition a tensor into segment of len `segment_len`
    along the first dimension. The tensor is padded with 0 in order
    to ensure that the first dimension is a multiple of `segment_len`.

    Examples:
    ```python
    >>> tensor = [[1, 2, 3], [4, 5, 6]]
    >>> segment_len = 2
    >>> pad_and_partition(tensor, segment_len)
    [[[1, 2], [4, 5]], [[3, 0], [6, 0]]]
    ````

    Parameters:
        tensor (tf.Tensor):
            Tensor of known fixed rank
        segment_len (int):
            Segment length.

    Returns:
        tf.Tensor:
            Padded and partitioned tensor.
    """
    tensor_size = tf.math.floormod(tf.shape(tensor)[0], segment_len)
    pad_size = tf.math.floormod(segment_len - tensor_size, segment_len)
    padded = tf.pad(tensor, [[0, pad_size]] + [[0, 0]] * (len(tensor.shape) - 1))
    split = (tf.shape(padded)[0] + segment_len - 1) // segment_len
    return tf.reshape(
        padded, tf.concat([[split, segment_len], tf.shape(padded)[1:]], axis=0)
    )


def pad_and_reshape(instr_spec, frame_length, F) -> Any:
    spec_shape = tf.shape(instr_spec)
    extension_row = tf.zeros((spec_shape[0], spec_shape[1], 1, spec_shape[-1]))
    n_extra_row = (frame_length) // 2 + 1 - F
    extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
    extended_spec = tf.concat([instr_spec, extension], axis=2)
    old_shape = tf.shape(extended_spec)
    new_shape = tf.concat([[old_shape[0] * old_shape[1]], old_shape[2:]], axis=0)
    processed_instr_spec = tf.reshape(extended_spec, new_shape)
    return processed_instr_spec
