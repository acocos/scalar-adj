#!/usr/bin/env python

'''
socal.py

Pairwise adjective scoring using pointwise intensity scores from SoCAL

Paper: https://www.mitpressjournals.org/doi/abs/10.1162/COLI_a_00049

Data: https://github.com/sfu-discourse-lab/SO-CAL/blob/master/Resources/dictionaries/English/

Scores pair (a_i, a_j) as SOCAL(a_j)-SOCAL(a_i)
'''

import os, sys
import gzip
import numpy as np

SOCAL = '../data/socal/adj_dictionary1.11.txt'

def read_socal_scores():
    '''
    Read socal pointwise intensity scores from file
    '''
    socal = {}
    with open(SOCAL, 'rU') as fin:
        for line in fin:
            w, sc = line.strip().split('\t')
            socal[w] = float(sc)
    return socal

def score(pairs):
    '''
    Given a set of adjective pairs, return a score for each pair (a_i,a_j) equivalent
    to |SOCAL(a_j)|-|SOCAL(a_i)|
    '''
    # read pointwise scores
    socal_scores = read_socal_scores()
    
    # generate pairwise scores
    scores = {p: 0.0 for p in pairs}
    for a_i, a_j in pairs:
        socal_i = socal_scores.get(a_i, None)
        socal_j = socal_scores.get(a_j, None)
        if socal_i is None or socal_j is None:
            scores[(a_i,a_j)] = None
            continue
        if np.sign(socal_i) != np.sign(socal_j):
            scores[(a_i,a_j)] = 0.
        else:
            scores[(a_i,a_j)] = np.abs(socal_j)-np.abs(socal_i)
    return scores

