# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import skimage


def center_crop(image: np.ndarray, crop_dim: tuple[int, int]) -> np.ndarray:
    """Take a central crop of `image` of size `crop_dim` in matching dimensions.
    
    The provided `image` is assumed to be of shape (H, W, C) or (W, H, C) and the
    `crop_dim` matches the dimensions layout of the image.
    """
    assert image.shape[0] >= crop_dim[0] and image.shape[1] >= crop_dim[1], (
        f'The image {image.shape} must be at least as large as crop_dim: {crop_dim}')
    crop_dim0_before = (image.shape[0] - crop_dim[0]) // 2
    crop_dim0_after  = (image.shape[0] - crop_dim[0]) // 2 + (image.shape[0] - crop_dim[0]) % 2
    crop_dim1_before = (image.shape[1] - crop_dim[1]) // 2
    crop_dim1_after  = (image.shape[1] - crop_dim[1]) // 2 + (image.shape[1] - crop_dim[1]) % 2
    return skimage.util.crop(image, crop_width=(
        (crop_dim0_before, crop_dim0_after), (crop_dim1_before, crop_dim1_after), (0, 0)), copy=True)


def compute_resize_scale(image: np.ndarray, min_len: int):
    """Computes scale factor to make each dimension of `image` >= `min_len`."""
    return max(1.0 * min_len / image.shape[0], 1.0 * min_len / image.shape[1])


def rescale_image(image: np.ndarray, min_len: int) -> np.ndarray:
    """Rescale provided `image` to have its smallest side be `min_len`.

    The provided `image` is assumed to be of shape (H, W, C) or (B, H, W, C).
    """
    assert len(image.shape) in (3, 4), f'Image must have 3 or 4 dimensions; given: {image.shape}'

    scale = compute_resize_scale(image, min_len)
    if len(image.shape) == 3:
        # Dimensions: (height, width, channels)
        return skimage.transform.rescale(image, (scale, scale, 1))
    elif len(image.shape) == 4:
        # Dimensions: (batch, height, width, channels)
        return skimage.transform.rescale(image, (1, scale, scale, 1))

def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    # First, scale the input image data from [0, 255] to [0, 1]
    scaled = image / 255.0
    # Then, subtract the ImageNet mean and divide by ImageNet stddev
    avg = [0.485, 0.456, 0.406]
    stddev = [0.229, 0.224, 0.225]

    # process each channel {R, G, B} individually
    for chan in range(3):
        scaled[:, :, chan] = (scaled[:, :, chan] - avg[chan]) / stddev[chan]
    return scaled
