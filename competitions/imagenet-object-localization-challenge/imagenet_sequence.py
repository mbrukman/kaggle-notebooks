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
"""Load ImageNet dataset and provide a `keras.utils.Sequence` for training,
validation, and testing.

Sample usage:

    images_with_class = load_training_images(images_dir, num_dirs, num_images_per_dir)
    imagenet_sequence = AlexNetImageNetSequence(images_with_class)
"""

from __future__ import annotations

import numpy as np
from tensorflow import keras
import matplotlib
import PIL

from collections import namedtuple
import glob
import os


ImageWithClass = namedtuple('ImageWithClass', 'image_path objclass')


def load_training_images(images_training_dir: str,
                         synset_map: dict[str, int],
                         num_dirs: int,
                         num_images_per_dir: int) -> list[ImageWithClass]:
    images_with_class = []

    count_dirs = 0
    for dir in glob.glob(os.path.join(images_training_dir, 'n*')):
        count_dirs += 1
        if count_dirs > num_dirs:
            break

        dir_name = dir.split(os.sep)[-1]
        image_class = synset_map[dir_name]

        count_images = 0
        for image in glob.glob(os.path.join(dir, '*.JPEG')):
            count_images += 1
            if count_images > num_images_per_dir:
                break

            images_with_class.append(ImageWithClass(image_path=image, objclass=image_class))

    return images_with_class
