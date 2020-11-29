"""my_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

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

#   def _info(self) -> tfds.core.DatasetInfo:
#     """Returns the dataset metadata."""
#     # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
#     return tfds.core.DatasetInfo(
#         builder=self,
#         description=_DESCRIPTION,
#         features=tfds.features.FeaturesDict({
#             # These are the features of your dataset like images, labels ...
#         }),
#         # If there's a common (input, target) tuple from the
#         # features, specify them here. They'll be used if
#         # `as_supervised=True` in `builder.as_dataset`.
#         supervised_keys=None,  # e.g. ('image', 'label')
#         homepage='https://dataset-homepage/',
#         citation=_CITATION,
#     )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    path_ = '/media/data_dump/hemant/hemant/nlp/pegasus/junk/pegasus/TF Create/tfrec.csv'
    return {
        'train': self._generate_examples(path=path_),
        'validation': self._generate_examples(path=path_),
        'test': self._generate_examples(path=path_),
    }

  def _generate_examples(self, path):
    with tf.io.gfile.GFile(path) as f:  # path to custom data
        for i, line in enumerate(f):
            source, target = line.split('\t')
            yield i, {
                    '_source': source,
                    '_target': target,
                        }

    
    # """Yields examples."""
    # # TODO(my_dataset): Yields (key, example) tuples from the dataset
    # yield 'key', {}
