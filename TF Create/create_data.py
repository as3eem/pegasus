import pandas as pd
import tensorflow as tf

save_path = "./pegasus/data/testdata/ami.tfrecords"


df = pd.read_csv('./tfrec.csv', index_col=None)
data = df.drop(['Unnamed: 0'],axis=1)

with tf.io.TFRecordWriter(save_path) as writer:
    for row in data.values:
        inputs, targets = row[:-1], row[-1]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "inputs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[inputs[0].encode('utf-8')])),
                    "targets": tf.train.Feature(bytes_list=tf.train.BytesList(value=[targets.encode('utf-8')])),
                }
            )
        )
        writer.write(example.SerializeToString())

print("TF Data has been published successfuly. Kindly check following route \n ", save_path)