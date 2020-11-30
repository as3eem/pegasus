"""ami dataset."""

import tensorflow_datasets as tfds
from . import ami


class AmiTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for ami dataset."""
  # TODO(ami):
  DATASET_CLASS = ami.Ami
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
      'validation': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
DL_EXTRACT_RESULT = ""


if __name__ == '__main__':
  tfds.testing.test_main()
