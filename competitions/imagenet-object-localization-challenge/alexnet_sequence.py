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


class AlexNetTrainingSequence(keras.utils.Sequence):
    def __init__(self, images_with_class: list[ImageWithClass],
                 data_sequence_class,
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
            image_array = image_transforms.rescale_image(image_array, min_len=256)
            image_array = image_transforms.center_crop(image_array, (256, 256))

            # Here's how the paper describes the approach during training:
            #
            #     The first form of data augmentation consists of generating
            #     image translations and horizontal reflections.
            #
            #     We do this by extracting random 224 × 224 patches (and their
            #     horizontal reflections) from the 256 × 256 images and training our
            #     network on these extracted patches. This increases the size of
            #     our training set by a factor of 2048, though the resulting
            #     training examples are, of course, highly interdependent. Without
            #     this scheme, our network suffers from substantial overfitting,
            #     which would have forced us to use much smaller networks.
            #
            # TODO(mbrukman): include the text here about the second form of
            # data augmentation.
            #
            # Here's how the paper describes the approach to testing:
            #
            #     At test time, the network makes a prediction by extracting
            #     five 224 × 224 patches (the four corner patches and the center
            #     patch) as well as their horizontal reflections (hence ten
            #     patches in all), and averaging the predictions made by the
            #     network’s softmax layer on the ten patches.
            #
            # TODO(mbrukman): implement data augmentation for training and testing.
            # TODO(mbrukman): horizontal flip: `np.fliplr(image_array)`
            image_array = image_transforms.center_crop(image_array, (227, 227))

            num_classes = 1000
            obj_class = keras.utils.to_categorical(image_with_class.objclass, num_classes)
            return image_array, obj_class

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
