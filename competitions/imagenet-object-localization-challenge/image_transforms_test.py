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

import unittest


# Adapted from `image_transforms.py` verbatim to avoid having to install deps.
def compute_resize_scale(image_shape: tuple[int, int], min_len: int):
    """Computes scale factor to make each dimension of `image` >= `min_len`."""
    return max(1.0 * min_len / image_shape[0], 1.0 * min_len / image_shape[1])


class ImageTransformsTest(unittest.TestCase):

    def testScaleBothLarger(self):
        image_shape = (200, 250)
        min_len = 100
        scale = compute_resize_scale(image_shape, min_len)
        new_image_shape = tuple([scale * dim for dim in image_shape])
        self.assertEqual(new_image_shape, (100, 125))

    def testScaleOneEqualOneLarger(self):
        image_shape = (200, 100)
        min_len = 100
        scale = compute_resize_scale(image_shape, min_len)
        new_image_shape = tuple([scale * dim for dim in image_shape])
        self.assertEqual(new_image_shape, (200, 100))

    def testScaleBothEqual(self):
        image_shape = (200, 200)
        min_len = 200
        scale = compute_resize_scale(image_shape, min_len)
        new_image_shape = tuple([scale * dim for dim in image_shape])
        self.assertEqual(new_image_shape, (200, 200))

    def testScaleOneEqualOneSmaller(self):
        image_shape = (200, 100)
        min_len = 200
        scale = compute_resize_scale(image_shape, min_len)
        new_image_shape = tuple([scale * dim for dim in image_shape])
        self.assertEqual(new_image_shape, (400, 200))

    def testScaleBothSmaller(self):
        image_shape = (100, 100)
        min_len = 200
        scale = compute_resize_scale(image_shape, min_len)
        new_image_shape = tuple([scale * dim for dim in image_shape])
        self.assertEqual(new_image_shape, (200, 200))


if __name__ == '__main__':
    unittest.main()
