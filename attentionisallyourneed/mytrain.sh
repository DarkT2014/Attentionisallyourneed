#!/usr/bin/env bash

DATA_DIR=/tmp/t2t_datagen/
PARAM_SET=base
MODEL_DIR=model_dir/model_subword_4096_$PARAM_SET

# SOURCE_DIR=./test_data/source_data
INPUT_VOCAB=$DATA_DIR/en_sub_word.vocab
TARGET_VOCAB=$DATA_DIR/zh_sub_word.vocab

cd ../models
export PYTHONPATH="$PYTHONPATH:{$PWD}"
pip install --user -r official/requirements.txt

python official/transformer/transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR  --input_vocab_file $INPUT_VOCAB --target_vocab_file $TARGET_VOCAB
    

    # --param_set=$PARAM_SET