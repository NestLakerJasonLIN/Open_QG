#!/usr/bin/env bash

ID=$1

python3 src/train_record.py \
--temp_pt_file data_$ID.pt