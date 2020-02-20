#!/usr/bin/env bash

ID=$1

python3 src/test.py \
--temp_pt_file data_$ID.pt \
#--test_on_train TEST_ON_TRAIN \
#--pred_file pred_train_$ID.txt \
#--gold_file gold_train_$ID.txt