"""my_dataset dataset."""

import tensorflow_datasets as tfds

# TODO(my_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(my_dataset): BibTeX citation
_CITATION = """
"""


class MyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "inputs":tf.string, 
            "targets":tf.string
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('inputs', 'targets'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(my_dataset): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('/content/part-00000-77c4055a-9b49-4d01-a25e-263f1e03198b.tfrecords.gz')

    # TODO(my_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'train_imgs'),
    }

#   def _generate_examples(self, path):
#     """Yields examples."""
#     # TODO(my_dataset): Yields (key, example) tuples from the dataset
#     for f in path.glob('*.jpeg'):
#       yield 'key', {
#           'image': f,
#           'label': 'yes',
#       }

  def _generate_examples(self, path):
    with tf.compat.v1.io.tf_record_iterator(path, options=None) as f:
      for i, line in enumerate(f):
        source, target = line.split(',')
        yield i, {
            'ipnuts': source,
            'targets': target,
        }
        
