import tensorflow as tf
import tensorflow_datasets as tfds



path = '/media/data_dump/hemant/hemant/nlp/pegasus/junk/pegasus/TF Create/tfrec.csv'
with tf.io.gfile.GFile(path) as f:  # path to custom data
    for i, line in enumerate(f):
        print(len(line.split('\t')))
        source, target = line.split('\t')
            # print(i)
        # yield i, {
        #             'source': source,
        #             'target': target,
        #                 }
