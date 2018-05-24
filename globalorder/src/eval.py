#!/usr/bin/env python

'''
eval.py

Perform evaluation of predicted vs gold standard adjective 
rankings, using metrics:
- Pairwise accuracy
- Kendall's Tau correlation coefficient
- Spearman's rho correlation coefficient
'''

import os, sys
import itertools
import numpy as np
from collections import Counter
from scipy.stats import spearmanr

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
    pairs = [(i,j) for i,j in itertools.product(words, words) if i<j]
    compared = {}
    for p in pairs:
        w1, w2 = p
        if ranks[w1] < ranks[w2]:
            cls = "<"
        elif ranks[w1] > ranks[w2]:
            cls = ">"
        else:
            cls = "="
        compared[p] = cls
    return compared

def pairwise_accuracy(pred_rankings, gold_rankings):
    g = []
    p = []
    for rankfile, predranks in pred_rankings.items():
        goldranks = gold_rankings.get(rankfile, None)
        if goldranks is None:
            sys.stderr.write('Could not find gold rankings for cluster %s\n' % rankfile)
            continue
        gold_compare = compare_ranks(goldranks)
        pred_compare = compare_ranks(predranks)
        if set(gold_compare.keys()) != set(pred_compare.keys()):
            sys.stderr.write('ERROR: Key mismatch for cluster %s\n' % rankfile)
            continue 
        pairs = sorted(list(gold_compare.keys()))
        for pr in pairs:
            g.append(gold_compare[pr])
            p.append(pred_compare[pr])
    correct = [1. if gg==pp else 0. for gg,pp in zip(g,p)]
    acc = sum(correct) / len(correct)
    return acc

def kendalls_tau(pred_rankings, gold_rankings, abs=False):
    taus = []
    ns = []
    for rankfile, predranks in pred_rankings.items():
        n_c = 0.
        n_d = 0.
        n = 0.
        ties_g = 0.
        ties_p = 0.
        goldranks = gold_rankings.get(rankfile, None)
        if goldranks is None:
            sys.stderr.write('Could not find gold rankings for cluster %s\n' % rankfile)
            continue
        words = sorted(goldranks.keys())
        for w_i in words:
            for w_j in words:
                if w_j >= w_i:
                    continue
                n += 1
                # check ties
                tied = False
                if goldranks[w_i]==goldranks[w_j]:
                    ties_g += 1
                    tied = True
                if predranks[w_i]==predranks[w_j]:
                    ties_p += 1
                    tied = True
                if tied:
                    continue
                # concordant/discordant
                dir_g = np.sign(goldranks[w_j]-goldranks[w_i])
                dir_p = np.sign(predranks[w_j]-predranks[w_i])
                if dir_g==dir_p:
                    n_c += 1
                else:
                    n_d += 1
        tau = (n_c - n_d) / np.sqrt((n-ties_g)*(n-ties_p))
        taus.append(tau)
        ns.append(n)
    taus = [t if not np.isnan(t) else 0. for t in taus]
    if abs:
        taus = [np.abs(t) for t in taus]
    tau_avg = np.average(taus, weights=ns)
    return tau_avg
    
def spearmans_rho_avg(pred_rankings, gold_rankings, abs=False):
    rhos = []
    ns = []
    for rankfile, predranks in pred_rankings.items():
        goldranks = gold_rankings.get(rankfile, None)
        if goldranks is None:
            sys.stderr.write('Could not find gold rankings for cluster %s\n' % rankfile)
            continue
        if set(goldranks.keys()) != set(predranks.keys()):
            sys.stderr.write('ERROR: Key mismatch for cluster %s\n' % rankfile)
            continue 
        ns.append(len(goldranks))
        words = sorted(goldranks.keys())
        predscores = [predranks[w] for w in words]
        goldscores = [goldranks[w] for w in words]
        r, p = spearmanr(predscores, goldscores)
        if np.isnan(r):
            r = 0.
        rhos.append(r)
    if abs:
        rho = np.average(np.abs(rhos), weights=ns)
    else:
        rho = np.average(rhos, weights=ns)
    return rho
    
def spearmans_rho(pred_rankings, gold_rankings, abs=False):
    g = []
    p = []
    for rankfile, predranks in pred_rankings.items():
        goldranks = gold_rankings.get(rankfile, None)
        if goldranks is None:
            sys.stderr.write('Could not find gold rankings for cluster %s\n' % rankfile)
            continue
        if set(goldranks.keys()) != set(predranks.keys()):
            sys.stderr.write('ERROR: Key mismatch for cluster %s\n' % rankfile)
            continue 
        words = sorted(goldranks.keys())
        predscores = [predranks[w] for w in words]
        goldscores = [goldranks[w] for w in words]
        g.extend(goldscores)
        p.extend(predscores)
    rho, pval = spearmanr(p, g)
    return rho, pval

if __name__ == "__main__":
    pred_dir = sys.argv[1]
    gold_dir = sys.argv[2]
    
    pred_rankings = read_rankings(pred_dir)
    gold_rankings = read_rankings(gold_dir)
    
    ## Pairwise accuracy
    print("Pairwise accuracy: %0.4f\n" % pairwise_accuracy(pred_rankings, gold_rankings))
    
    ## Kendall's tau
    print("Kendall's Tau (AVG): %0.4f\n" % kendalls_tau(pred_rankings, gold_rankings, abs=False))
#     print("Kendall's Tau (AVG ABS): %0.4f\n" % kendalls_tau(pred_rankings, gold_rankings, abs=True))
    
    ## Spearman's rho
    print("Spearman's Rho: %0.4f (%0.4f)\n" % spearmans_rho(pred_rankings, gold_rankings, abs=False))
