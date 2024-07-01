#!/bin/bash

echo "BEAM SIZE=$1"
echo "MODEL=$2"
BATCH=1000
INDEX=$1
MODEL=$2
DATA_PATH=./data/iwslt14/iwslt14.tokenized.de-en/

# sl=en
# tl=de

sl=en
tl=de

python translate.py -gpu 8 -fp32 -batch_size $BATCH -src $DATA_PATH/test.$sl -replace_unk -alpha 1 -beta 0.0 -length_penalty wu -coverage_penalty wu -output $DATA_PATH/test.hyp$INDEX.$tl -beam_size $INDEX -min_length 1  -model $MODEL

sed -r 's/(@@ )|(@@ ?$)//g' $DATA_PATH/test.hyp$INDEX.$tl | perl multi-bleu-detok.perl $DATA_PATH/tmp/test.$tl 
