#!/usr/bin/env bash

###################################
# adjective ordering experiments
###################################

MODELFILE=../../pairpredict/data/trainval_jjgraph/models/model.c0.0001.adverb_binary.elasticnet.nobias.trainval_jjgraph.pkl

# generate weights
EXPERIMENT=globalranking
mkdir -p ../data/weightacc
for TESTSET in demelo crowd wilkinson;
do
    mkdir -p ../data/$TESTSET/$EXPERIMENT/weights
    for METHOD in patterns rblr socal combined2_socal_patterns combined2_socal_rblr combined2_patterns_rblr combined3_patterns_socal_rblr combined3_patterns_rblr_socal combined3_socal_rbr_patterns combined3_socal_patterns_rblr combined3_rblr_socal_patterns combined3_rblr_patterns_socal;
    do
        echo $TESTSET $EXPERIMENT $METHOD
        python score_pairwise.py -d ../data/$TESTSET/terms -m $METHOD -o ../data/$TESTSET/$EXPERIMENT/weights/$METHOD.weights -M $MODELFILE
        python weights_acc.py -g ../data/$TESTSET/gold_rankings -w ../data/$TESTSET/$EXPERIMENT/weights/$METHOD.weights > ../data/weightacc/$EXPERIMENT.$TESTSET.$METHOD.weightacc
    done
done

# run ordering 
for TESTSET in demelo crowd wilkinson; 
do
    for WGT in patterns rblr socal combined2_socal_patterns combined2_socal_rblr combined2_patterns_rblr combined3_patterns_socal_rblr combined3_patterns_rblr_socal combined3_socal_rbr_patterns combined3_socal_patterns_rblr combined3_rblr_socal_patterns combined3_rblr_patterns_socal;
    do
        OUTDIR=../data/$TESTSET/$EXPERIMENT/pred\_rankings\_$WGT
        mkdir -p $OUTDIR
        for f in `ls ../data/$TESTSET/terms`;
        do
            python order_milp.py -t ../data/$TESTSET/terms/$f -w ../data/$TESTSET/$EXPERIMENT/weights/$WGT.weights -o $OUTDIR/${f%.terms}.rankings
        done
    done
done

# score global ordering
mkdir -p ../data/rankresults
for TESTSET in demelo crowd wilkinson;
do
    for WGT in patterns rblr socal combined2_socal_patterns combined2_socal_rblr combined2_patterns_rblr combined3_patterns_socal_rblr combined3_patterns_rblr_socal combined3_socal_rbr_patterns combined3_socal_patterns_rblr combined3_rblr_socal_patterns combined3_rblr_patterns_socal;
    do
        echo $TESTSET $WGT
        python eval.py ../data/$TESTSET/$EXPERIMENT/pred\_rankings\_$WGT ../data/$TESTSET/gold_rankings > ../data/rankresults/$EXPERIMENT.$TESTSET.$WGT.scores
    done
done