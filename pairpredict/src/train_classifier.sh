#!/usr/bin/env bash

## generate features
TRAINSPLIT=trainval_jjgraph
TRAINDIR=../data/$TRAINSPLIT
mkdir -p $TRAINDIR/models
mkdir -p $TRAINDIR/predictions

python create_allgraph_trainset.py > $TRAINDIR/labels
cat $TRAINDIR/labels | python adverb_binary_features.py ../data/rbmin10.txt > $TRAINDIR/adverb_binary

## cross-validate
# python classify.py -d $TRAINDIR -f adverb_binary -t -i adverb_binary.elasticnet -n 10

## train classifier
C=0.0001
FEATURE=adverb_binary
NM=c$C.$FEATURE.elasticnet.nobias.$TRAINSPLIT
MODELFILE=$TRAINDIR/models/model.$NM.pkl
python classify.py -d $TRAINDIR -f $FEATURE -i $NM -m $MODELFILE -v -c $C
