#!/usr/bin/env python

'''
scalar_pairs.py

Make predictions for IQAP dataset using scalar pairs scoring
methods

This code is only slightly modified from the original code of deMarneffe and Potts:
https://github.com/cgpotts/iqap/tree/master/ACL2010
'''
import csv
import os, sys
from scalars import AnnotatedDialogues, Dialogue
import numpy as np
import pickle

import score_pairwise
import rb_lr
import patterns
import socal

DATAFILE = '../data/indirect-answers.combined.imdb-predictions.csv'
MDIR = '../../pairpredict/data/trainval_jjgraph/models'
RBMODELFILE = os.path.join(MDIR, 'model.c0.0001.adverb_binary.elasticnet.nobias.trainval_jjgraph.pkl')

def decision(dialogue, scores):
    '''
    Compute a dialogue polarity ('yes','no','uncertain')
    based on word-level scores
    '''
    classification = dialogue.classification
    negation = dialogue.negation
    modsQ = dialogue.modsQ
    modsA = dialogue.modsA
    adv = dialogue.adverb
    if classification == "avoided_adjective.txt":
        # modsQ[0] will be "JJ NN"
        modQstr = modsQ[0].split(" ")[0]
        val = scores.get(modQstr, None)
        if modQstr == None:
            return (modQstr, '', 0, "uncertain")
        else: 
            if val < 0:
                return (modQstr, '', val, "no")
            else:
                return (modQstr, '', val, "yes")
    else:        
        for modQstr in modsQ:
            for modAstr in modsA:
                modQval = scores.get(modQstr, None)
                modAval = scores.get(modAstr, None)
                if modQval == None or modAval == None:
                    return (modQstr, modAstr, 'missing jj', "uncertain")
                elif modQstr == modAstr:
                    return (modQstr, modAstr, 0, reverse_prediction("yes", negation))
                elif np.sign(modQval) != np.sign(modAval):
                    return (modQstr, modAstr, -1000, reverse_prediction("no", negation))
                elif abs(modQval) <= abs(modAval):
                    return (modQstr, modAstr, abs(modAval) - abs(modQval), reverse_prediction("yes", negation))
                elif abs(modQval) > abs(modAval):
#                     ## This modification produces better results
#                     if adv:
#                         return (modQstr, modAstr, abs(modAval) - abs(modQval), reverse_prediction("yes", negation))
#                     else:
#                         return (modQstr, modAstr, abs(modAval) - abs(modQval), reverse_prediction("no", negation))
                    return (modQstr, modAstr, abs(modQval) > abs(modAval), reverse_prediction("no", negation))
        return (modQstr, modAstr, 0, "uncertain")


def predict_dialogue(dialogue, scores):
    '''
    Compute a dialogue polarity ('yes','no','uncertain')
    based on pairwise scores
    '''
    classification = dialogue.classification
    negation = dialogue.negation
    modsQ = dialogue.modsQ
    modsA = dialogue.modsA
    adv = dialogue.adverb
    if classification == "avoided_adjective.txt":
        # TODO: Update
        return (modsQ[0], '', 'na', "yes")
    else:
        for modQstr in modsQ:
            for modAstr in modsA:
                modAstr = modAstr.lower()
                modQstr = modQstr.lower()
                sc = scores[(modQstr, modAstr)]
                if modQstr==modAstr:
                    return (modQstr, modAstr, 'same jj', reverse_prediction("yes", negation))
                elif sc == 0:
                    return (modQstr, modAstr, sc, reverse_prediction("no", negation))
                elif sc > 0:
                    return (modQstr, modAstr, sc, reverse_prediction("yes", negation))
                elif sc < 0 and sc is not None:
                    return (modQstr, modAstr, sc, reverse_prediction("no", negation))
        return (modQstr, modAstr, 'default', "uncertain")

def reverse_prediction(prediction, negation):        
    if prediction == "yes" and negation != "":
        return "no"
    elif prediction == "no" and negation != "":
        return "yes"
    else:
        return prediction


if __name__=="__main__":
    
    method = sys.argv[1]  # one of (patterns, rblr, socal)
    outfile = sys.argv[2]
    
    # read in data
    data = AnnotatedDialogues(DATAFILE)
    pairs = []
    for d in data.dialogues:
        for modQ in d.modsQ:
            for modA in d.modsA:
                pairs.append((modQ, modA))
                pairs.append((modA, modQ))
    
    # generate scores
    if method == 'patterns':
        scores = score_pairwise.scale_scores(patterns.score(pairs))
    elif method == 'rblr':
        scores = score_pairwise.scale_scores(rb_lr.score(pairs, RBMODELFILE))
    elif method == 'socal': 
        scores_pointwise = socal.score(pairs)
    
    # predict for QA pairs
    results = []
    for dialogue in data.dialogues:
        if method == 'socal':
            qa, aa, sc, pred = decision(dialogue, scores_pointwise)
        else:
            qa, aa, sc, pred = predict_dialogue(dialogue, scores)
        
        act = dialogue.tri_dominant_answer
        hitid = dialogue.hitid
        results.append({'hitid': dialogue.hitid, 
                        'prediction': pred, 
                        'actual': act, 
                        'classification': dialogue.classification, 
                        'negation': dialogue.negation,
                        'modQ': qa,
                        'modA': aa, 
                        'score': sc})
    
    # output results and evaluate
    with open(outfile, 'w') as fout:
        headers = ['hitid','prediction','actual','classification','negation','modQ','modA','score']
        wrt = csv.DictWriter(fout, fieldnames=headers)
        wrt.writeheader()
        for res in results:
            wrt.writerow(res)
        