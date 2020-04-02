#!/usr/bin/env bash

ID=$1
EPOCH=$2

python3 src/preprocess.py \
--temp_pt_file data_$ID.pt \
--model_statistics_file model_statistics_$ID.pt \
--checkpoint_file checkpoint_$ID.pt \
--pred_file pred_$ID.txt \
--gold_file gold_$ID.txt \
--print_params PRINT_PARAMS \
--num_epochs $EPOCH \
--d_model 332
#--beam_size 1
#--no_copy
#--params.d_model 128 \
#--params.num_heads 1 \
#--params.d_k 64 \
#--params.dropout 0.5 \
#--params.num_layers 2 \
#--params.num_epochs $EPOCH
