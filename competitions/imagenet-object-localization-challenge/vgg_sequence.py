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

from imagenet_sequence import ImageWithClass
import image_transforms

from tensorflow import keras
import PIL
import matplotlib
import numpy as np

from typing import Type


def subtractImageNetAverage(image: np.ndarray) -> np.ndarray:
    # Subtract the ImageNet mean
    imagenet_avg = [0.485, 0.456, 0.406]

    normalized = np.empty(image.shape)
    for chan in range(3):
        normalized[:, :, chan] = image[:, :, chan] - imagenet_avg[chan]
    return normalized


class VGGTrainingSequence(keras.utils.Sequence):
    def __init__(self, images_with_class: list[ImageWithClass],
                 data_sequence_class: Type,
                 batch_size: int = 128):
        super().__init__()
        self.images_with_class = images_with_class
        self.batch_size = batch_size
        self.data_sequence = data_sequence_class(num_items=len(images_with_class),
                                                 batch_size=batch_size)

    def __len__(self):
        return len(self.data_sequence)

    def _load_image(self, image_with_class: ImageWithClass) -> tuple[np.ndarray, np.ndarray]:
        with PIL.Image.open(image_with_class.image_path) as pil_image:
            pil_image_rgb = pil_image if pil_image.mode == 'RGB' else pil_image.convert('RGB')
            image_array = matplotlib.image.pil_to_array(pil_image_rgb)

            # TODO(mbrukman): include details of training process from the paper.
            image_array = image_transforms.rescale_image(image_array, min_len=224)
            image_array = image_transforms.center_crop(image_array, (224, 224))
            # image_array = subtractImageNetAverage(image_array)

            #num_classes = 1000
            objclass = image_with_class.objclass
            #objclass = keras.utils.to_categorical(image_with_class.objclass, num_classes)
            return image_array, objclass

    def __getitem__(self, index: int):
        images = []
        obj_classes = []

        _, positions = self.data_sequence[index]
        for pos in positions:
            image, obj_class = self._load_image(self.images_with_class[pos])
            images.append(image)
            obj_classes.append(obj_class)

        return np.array(images), np.array(obj_classes)

    def on_epoch_end(self):
        self.data_sequence.on_epoch_end()
