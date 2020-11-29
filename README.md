# PEGASUS

Pre-training with Extracted Gap-sentences for Abstractive SUmmarization
Sequence-to-sequence models, or PEGASUS, uses self-supervised objective Gap
Sentences Generation (GSG) to train a transformer encoder-decoder model. The
paper can be found on [arXiv](https://arxiv.org/abs/1912.08777). ICML 2020 accepted.

# Setup

## install library and dependencies

Clone library on github and install requirements.

```
git clone https://github.com/google-research/pegasus
cd pegasus
export PYTHONPATH=.
pip3 install -r requirements.txt
```

Download vocab, pretrained and fine-tuned checkpoints of all experiments from [Google Cloud](https://console.cloud.google.com/storage/browser/pegasus_ckpt).

Alternatively in terminal, follow the instruction and install [gsutil](https://cloud.google.com/storage/docs/gsutil_install). Then

```
mkdir ckpt
gsutil cp -r gs://pegasus_ckpt/ ckpt/

```

# Finetuning on downstream datasets

## on existing dataset

Finetune on an existing dataset `aeslc`.

```
python3 pegasus/bin/train.py --params=aeslc_transformer \
--param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model \
--train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 \
--model_dir=ckpt/pegasus_ckpt/aeslc
```

Fine Tune on AMI Dataset

```
python3 pegasus/bin/evaluate.py --params=ami_transformer \
--param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 \
--model_dir=ckpt/pegasus_ckpt/
```



If you would like to finetune on a subset of dataset, please refer to the [example of input pattern](https://github.com/google-research/pegasus/blob/master/pegasus/data/datasets.py#L186).

Evaluate on the finetuned dataset.

```
python3 pegasus/bin/evaluate.py --params=aeslc_transformer \
--param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 \
--model_dir=ckpt/pegasus_ckpt/aeslc
```

Note that the above example is using a single GPU so the batch_size is much smaller
than the results reported in the paper.

## add new finetuning dataset

Two types of dataset format are supported: [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets) or TFRecords.

[This tutorial](https://www.tensorflow.org/datasets/add_dataset) shows how to add a new dataset in TFDS.
(The fine-tuning dataset is expected to be supervised, please provide
`supervised_keys` in dataset info).

Tfrecords format requires each record to be a tf example of `{"inputs":tf.string, "targets":tf.string}`.

For example, if you registered a TFDS dataset called `new_tfds_dataset` for training and evaluation, and have some files in tfrecord format called `new_dataset_files.tfrecord*` for test, they can be registered in `/pegasus/params/public_params.py`.

```
@registry.register("new_params")
def my_param(param_overrides):
  return public_params.transformer_params(
      {
          "train_pattern": "tfds:new_tfds_dataset,train",
          "dev_pattern": "tfds:new_tfds_dataset,validation",
          "test_pattern": "tfrecord:new_dataset_files.tfrecord*",
          "max_input_len": 512,
          "max_output_len": 128,
          "train_steps": 10000,
          "learning_rate": 0.0001,
          "batch_size": 8,
      }, param_overrides)
```

## Evaluation metrics.

Evaluation results can be found in `mode_dir`. Summarization metrics are automatically
calculated for each evaluation point.

-   [ROUGE](https://www.aclweb.org/anthology/W04-1013.pdf) is the main metric
    for summarization quality.

-   [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf) is an alternative
    quality metric for language generation.

-   [Extractive Fragments Coverage & Density](https://arxiv.org/pdf/1804.11283.pdf)
    are metrics that measures the abstractiveness of the summary.


-   Repetition Rates measures generation repetition failure modes.

-   Length statistics measures the length distribution of decodes comparing to gold summary.

Several types of output files can be found in `model_dir`

-   text_metrics-*.txt: above metrics in text format. Each row contains metric
    name, 95% lower bound value, mean value, 95% upper bound value.
-   inputs-*.txt, targets-*.txt, predictions-*.txt: raw text files of model
    inputs/outputs.


# Pre-training

Pretraining (on C4 or any other corpus) requires a customly built tensorflow that includes ops for on-the-fly parsing that processes raw text document into model inputs and targets ids. Please refer to pegasus/ops/pretrain_parsing_ops.cc and pegasus/data/parsers.py for details.

If you use this code or these models, please cite the following paper:
```
@misc{zhang2019pegasus,
    title={PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization},
    author={Jingqing Zhang and Yao Zhao and Mohammad Saleh and Peter J. Liu},
    year={2019},
    eprint={1912.08777},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```



