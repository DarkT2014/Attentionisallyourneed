#!/usr/bin/env bash

PROBLEM=translate_enzh_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu


DATA_DIR=/tmp/t2t_datagen

t2t-datagen --data_dir=$DATA_DIR --tmp_dir=$DATA_DIR --problem=$PROBLEM