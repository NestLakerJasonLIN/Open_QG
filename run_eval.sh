#!/usr/bin/env bash

ID=$1

python3 evaluate/eval.py \
--pred_file pred_$ID.txt \
--gold_file gold_$ID.txt
