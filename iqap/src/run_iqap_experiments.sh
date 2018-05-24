#!/usr/bin/env bash

## IQAP experiments

mkdir -p ../results

for WGT in patterns rblr socal;
do
    RESULTFILE=../results/$WGT\_results.elasticnet.csv
    python scalar_pairs.py $WGT $RESULTFILE
done

# score (individual weights, and combined)
python score.py ../data/indirect-answers.combined.imdb-predictions.csv ../results > ../allresults.csv