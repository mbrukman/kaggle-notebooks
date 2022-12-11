#!/usr/bin/python
#
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
"""Provides functions to map from directory names (also class names) to int.

Sample usage:

    synset_mapping = compute_synset_mapping()
    test_synset_mapping(synset_mapping)
"""

from __future__ import annotations

import re
import os
from typing import Dict


DATA_ROOT = '/kaggle/input/imagenet-object-localization-challenge'

def compute_synset_mapping(root_dir: str = DATA_ROOT) -> dict[str, int]:
    """Creates a mapping of string class name to numeric id."""
    synset_mapping: dict[str, int] = {}
    with open(os.path.join(root_dir, 'LOC_synset_mapping.txt')) as synset_mapping_file:
        counter = 0
        for line in synset_mapping_file.readlines():
            synclass = re.sub(r'^(n[0-9]+)\s.+$', r'\1', line.strip())
            synset_mapping[synclass] = counter
            counter += 1

    return synset_mapping


def test_synset_mapping(mapping: Dict[str, int]) -> None:
    """Validate synset mapping by comparing with TensorFlow ImageNet loader.

    A few sample class name -> numeric ID mappings are provided on this page:
    https://www.tensorflow.org/datasets/catalog/imagenet2012 .
    """
    tf_imagenet_df = dict(
      n02097047=196,
      n01682714=40,
      n03134739=522,
      n04254777=806,
      n02859443=449,
      n02096177=192,
      n02107683=239,
      n01443537=1,
      n02264363=318,
      n03759954=650,
    )

    for synclass, label in tf_imagenet_df.items():
        assert synset_mapping[synclass] == label, f'Error: we map {synclass} to {synset_mapping[synclass]} while TensorFlow to {label}'
