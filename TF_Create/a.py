# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Annotated Enron Subject Line Corpus Dataset."""


import tensorflow_datasets as tfds
import tensorflow as tf


class Ami(tfds.core.GeneratorBasedBuilder):
  """Annotated Enron Subject Line Corpus Dataset."""

  VERSION = tfds.core.Version("1.0.0")

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            '_DOCUMENT': tfds.features.Text(),
            '_SUMMARY': tfds.features.Text()
        }),
    )

  def _split_generators(self, dl_manager):
    """Download the data and define splits."""
    path_ = '/media/data_dump/hemant/hemant/nlp/pegasus/junk/pegasus/TF_Create/'
    return {
           'validation': self._generate_examples(path=os.path.join(path_,"val.csv")),
           'train': self._generate_examples(path= os.path.join(path_,"train.csv")),
           'test': self._generate_examples(path=os.path.join(path_,"test.csv")),
              }

    

  def _generate_examples(self, path):
    with tf.io.gfile.GFile(path) as f:  # path to custom data
        for i, line in enumerate(f):
            source, target = line.split('\t')
            yield i, {'_DOCUMENT': source, '_SUMMARY': target}
