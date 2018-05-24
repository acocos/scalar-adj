'''
weights_acc.py

Given a set of gold standard rankings, and a set of 
generated pairwise weights, calculate:
- The coverage of the generated weighted pairs against all
    comparable pairs in the gold rankings
- The directional accuracy of the generated weighted pairs
    (i.e. if weight for (w1,w2) is positive, then (w1,w2) should
    be weak-strong)
'''

import os, sys
import argparse
import numpy as np
import itertools


def load_weights(wf):
    weights = {}
    with open(wf, 'rU') as fin:
        for line in fin:
            w1, w2, wgt = line.strip().split('\t')
            weights[(w1,w2)] = float(wgt)
    return weights

def read_rankings(dirname):
    '''
    ::param: dirname : directory holding results files
    ::returns: dict  : {word: score}
    '''
    rankingfiles = os.listdir(dirname)
    results = {}
    for rf in rankingfiles:
        results[rf] = {}
        with open(os.path.join(dirname, rf), 'rU') as fin:
            for line in fin:
                score, words = line.strip().split('\t')
                words = words.split(' || ')
                score = float(score)
                for word in words:
                    results[rf][word] = score
    return results

def compare_ranks(ranks):
    words = ranks.keys()
    pairs = [(i,j) for i,j in itertools.product(words, words)]
    compared = {}
    for p in pairs:
        w1, w2 = p
        if w1==w2:
            cls = 'X'
        elif ranks[w1] < ranks[w2]:
            cls = "<"
        elif ranks[w1] > ranks[w2]:
            cls = ">"
        else:
            cls = "="
        compared[p] = cls
    return compared

def compare_weights(words, wgts):
    pairs = [(i,j) for i,j in itertools.product(words, words)]
    compared = {}
    for p in pairs:
        w1, w2 = p
        if w1==w2:
            cls = 'X'
        else:
            wgt = wgts.get(p, None)
            if wgt is None or wgt==0:
                cls = '0'
            elif wgt > 0:
                cls = '<'
            elif wgt < 0:
                cls = '>'
        compared[p] = cls
    return compared

def srtd(dct, rev=False):
    return sorted(dct, key=dct.get, reverse=rev)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pairwise adjective scoring.')
    parser.add_argument('-g', '--golddir', type=str, dest='golddir',
                        help='Directory containing gold standard rankings')
    parser.add_argument('-w', '--weightfile', type=str, dest='weightfile',
                        help='File containing pairwise weights')
    args = parser.parse_args()
    
    gold_ranks = read_rankings(args.golddir) 
    predweights = load_weights(args.weightfile) 
    
    npairs = []
    nmatches = []
    pmatches = []
    
    for jjset in gold_ranks:
        goldranks = gold_ranks[jjset]
        gold = compare_ranks(goldranks)
        orderedwords = srtd(goldranks)
        pred = compare_weights(orderedwords, predweights)
        numpairs = len([pr for pr,dr in gold.items() if dr!='X'])
        nummatches = len([pr for pr,dr in pred.items() if dr!='X' and dr!='0'])
        posmatches = len([pr for pr,dr in pred.items() if dr!='X' and dr!='0' and dr==gold[pr]])
        try:
            ovlp = float(nummatches)/numpairs
        except ZeroDivisionError:
            ovlp = 0.
        try:
            acc = float(posmatches)/nummatches
        except ZeroDivisionError:
            acc = np.nan
        print jjset, 'Overlap: %0.3f    Acc: %0.3f' % (ovlp, acc)
        npairs.append(numpairs)
        nmatches.append(nummatches)
        pmatches.append(posmatches)
    ovlp = float(sum(nmatches))/sum(npairs)
    accnum, accden = zip(*[(p,n) for (p,n) in zip(pmatches, nmatches) if nmatches>0])
    acc = float(sum(accnum))/sum(accden)
    print '*************************'
    print 'Overall results:'
    print '  Overlap: %0.3f   Acc: %0.3f' % (ovlp, acc)
    