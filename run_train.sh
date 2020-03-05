#!/usr/bin/env bash

ID=$1

python3 src/train_jadore.py \
--temp_pt_file data_$ID.pt \
--tensorboard_dir $ID
