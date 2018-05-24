#!/usr/bin/env python

'''
patterns.py

Implementation of pairwise adjective scoring from DeMelo & Bansal 2013.

Uses frequency of pre-determined 'Strong-Weak' and 'Weak-Strong' patterns
extracted from the Google NGram dataset
'''

import os, sys
import gzip

UNIGRAM_COUNTS = '../../globalorder/data/pattern_data/1gms.vocab_cs.gz'
PAT_COUNTS_SW = '../../globalorder/data/pattern_data/paths.sw.counts'
PAT_COUNTS_WS = '../../globalorder/data/pattern_data/paths.ws.counts'
INST_COUNTS_SW = '../../globalorder/data/pattern_data/word-paths.sw.counts'
INST_COUNTS_WS = '../../globalorder/data/pattern_data/word-paths.ws.counts'
MINCNT = 100

def flatten(l):
    return [item for sublist in l for item in sublist]

def grab_unigram_counts(vocab):
    counts = {}
    with gzip.open(UNIGRAM_COUNTS, 'r') as fin:
        for line in fin:
            w, c = line.strip().split('\t')
            if w in vocab:
                counts[w] = int(c)
    return counts

def read_pattern_counts(patternfile):
    counts = {}
    with open(patternfile, 'rU') as fin:
        for line in fin:
            p, c = line.strip().split('\t')
            counts[p] = int(c)
    return counts

def read_inst_counts(instfile, pairset):
    counts = {}
    with open(instfile, 'rU') as fin:
        for line in fin:
            pat, w1, w2, c = line.strip().split('\t')
            c = int(c)
            if pat not in counts:
                counts[pat] = {}
            if (w1, w2) in pairset:
                counts[pat][(w1, w2)] = c
    return counts

def score(pairs):
    '''
    
    '''
    vocab = set(flatten(pairs))
    # read unigram, path, and instance counts
    unigram_counts = grab_unigram_counts(vocab)
    pattern_counts = {'SW': read_pattern_counts(PAT_COUNTS_SW),
                      'WS': read_pattern_counts(PAT_COUNTS_WS)}
    inst_counts = {'SW': read_inst_counts(INST_COUNTS_SW, pairs),
                   'WS': read_inst_counts(INST_COUNTS_WS, pairs)}
    P_sw = float(sum(pattern_counts['SW'].values()))
    P_ws = float(sum(pattern_counts['WS'].values()))
    
    # generate pairwise scores
    scores = {p: 0.0 for p in pairs}
    for pat, patdct in inst_counts['SW'].items():
        for pr, cnt in patdct.items():
            a1, a2 = pr
            scores[(a1,a2)] -= cnt / P_sw
            scores[(a2,a1)] += cnt / P_sw
    for pat, patdct in inst_counts['WS'].items():
        for pr, cnt in patdct.items():
            a1, a2 = pr
            scores[(a1,a2)] += cnt / P_ws
            scores[(a2,a1)] -= cnt / P_ws
    normscores = {}
    for pr, sc in scores.items():
        a1, a2 = pr
        if a1 not in unigram_counts or a2 not in unigram_counts:
            normscores[(a1,a2)] = None
            continue
        sc /= (unigram_counts.get(a1, MINCNT) * unigram_counts.get(a2, MINCNT))
        normscores[(a1,a2)] = sc
    return normscores
        