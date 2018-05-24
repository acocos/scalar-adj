#!/usr/bin/env python

'''
score_pairwise.py

Generate pairwise intensity scores for adjective pairs.

Choose from three methods:
- Pattern-based score (DeMelo & Bansal 2013)
- Adverb-based score
- Combination of the two

'''

import os, sys
import argparse
from collections import Counter
import numpy as np

import rb_lr
import patterns
import socal

def read_pairs(dirname):
    '''
    Read all comparable pairs from the same scale from scale term files in dirname
    Returns: a set of 2-tuples
    '''
    termsfiles = [os.path.join(dirname, f) for f in os.listdir(dirname)]
    pairs = set()
    for tf in termsfiles:
        words = set()
        with open(tf, 'rU') as fin:
            for line in fin:
                __, w = line.strip().split('\t')
                words.add(w)
        for w1 in words:
            for w2 in words:
                if w1!=w2:
                    pairs.add((w1, w2))
    return pairs

def read_pairs_scales(dirname):
    '''
    Read all comparable pairs from the same scale from scal term files in dirname;
    keep track of which file each pair comes from
    Returns: a dict of {filename: set([(w1,w2),...])}
    '''
    termsfiles = [os.path.join(dirname, f) for f in os.listdir(dirname)]
    pairs = {}
    for tf in termsfiles:
        tf_clean = os.path.basename(tf).replace('.terms','')
        filepairs = set()
        words = set()
        with open(tf, 'rU') as fin:
            for line in fin:
                __, w = line.strip().split('\t')
                words.add(w)
        for w1 in words:
            for w2 in words:
                if w1!=w2:
                    filepairs.add((w1, w2))
        pairs[tf_clean] = filepairs
    return pairs


def scale_factor(wgtdct):
    scaling_factor = sys.maxint
    for v in wgtdct.values():
        if v is None:
            continue
        v = abs(v)
        if v > 0 and v < scaling_factor:
            scaling_factor = v
    if v == sys.maxint:
        scaling_factor = 1.
    else:
        scaling_factor = 1. / scaling_factor
    return scaling_factor
    
def scale_scores(wgtdct):
    sf = scale_factor(wgtdct)
    return {k: sf*v if v else None for k,v in wgtdct.items()}

def get_log_mean_std(weights):
    wgts = np.array([np.log(v) for v in weights.values() if v>0])
    return np.mean(wgts), np.std(wgts)

def combine_scores(m1, m2, m1_mean=0., m1_std=1., m2_mean=0., m2_std=1., offset=5.):
    if len(m1) == 0:
        return m2
    if len(m2) == 0:
        return m1
    final_scores = {}
    for k, m1wgt in m1.items():
        if k in m2:
            m2wgt = m2[k]
            if m1wgt == 0 or m1wgt is None:
                norm_m1wgt = 0.
            else:
                norm_m1wgt = np.sign(m1wgt) * ((np.log(np.abs(m1wgt)) - m1_mean) / m1_std + offset)
            if m2wgt == 0 or m2wgt is None:
                norm_m2wgt = 0.
            else:
                norm_m2wgt = np.sign(m2wgt) * ((np.log(np.abs(m2wgt)) - m2_mean) / m2_std + offset)
            ## default to m2 
            if np.abs(norm_m1wgt) > 0:
                final_scores[k] = norm_m1wgt
            elif np.abs(norm_m2wgt) > 0:
                final_scores[k] = norm_m2wgt
            else:
                final_scores[k] = 0.
    return final_scores

def combine_three_scores(m1, m2, m3, m1_mean=0., m1_std=1., m2_mean=0., m2_std=1., 
                         m3_mean=0., m3_std=1., offset=5.):
    final_scores = {}
    for k, m1wgt in m1.items():
        if k in m2 and k in m3:
            m2wgt = m2[k]
            m3wgt = m3[k]
            if m1wgt == 0 or m1wgt is None:
                norm_m1wgt = 0.
            else:
                norm_m1wgt = np.sign(m1wgt) * ((np.log(np.abs(m1wgt)) - m1_mean) / m1_std + offset)
            if m2wgt == 0 or m2wgt is None:
                norm_m2wgt = 0.
            else:
                norm_m2wgt = np.sign(m2wgt) * ((np.log(np.abs(m2wgt)) - m2_mean) / m2_std + offset)
            if m3wgt == 0 or m3wgt is None:
                norm_m3wgt = 0.
            else:
                norm_m3wgt = np.sign(m3wgt) * ((np.log(np.abs(m3wgt)) - m3_mean) / m3_std + offset)
            if np.abs(norm_m1wgt) > 0:
                final_scores[k] = norm_m1wgt
            elif np.abs(norm_m2wgt) > 0:
                final_scores[k] = norm_m2wgt
            elif np.abs(norm_m3wgt) > 0:
                final_scores[k] = norm_m3wgt
            else:
                final_scores[k] = 0.
    return final_scores


def print_scores(scdct, fout):
    for pr, sc in Counter(scdct).most_common():
        w1, w2 = pr
        if sc is None:
            sc = 0.
        print >> fout, '\t'.join((w1, w2, str(sc)))
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pairwise adjective scoring.')
    parser.add_argument('-d', '--datadir', type=str, dest='datadir',
                        help='Directory containing terms files for generating pairwise scores')
    parser.add_argument('-m', '--method', default='combined2_patterns_rblr', const='combined2_patterns_rblr',
                        nargs='?', choices=['rblr', 'patterns', 'socal', 
                                            'combined2_rblr_patterns', 'combined2_rblr_socal', 
                                            'combined2_patterns_rblr', 'combined2_patterns_socal', 
                                            'combined2_socal_rblr', 'combined2_socal_patterns',
                                            'combined3_pat_socal_pp', 'combined3_pp_socal_pat',
                                            'combined3_pat_pp_socal', 'combined3_socal_pat_pp',
                                            'combined3_socal_pp_pat', 'combined3_pp_pat_socal'],
                        help='Method rblr, patterns, socal, or combined (default: %(default)s)')
    parser.add_argument('-o', '--outfile', type=str, default=None)
    parser.add_argument('-w', '--minweight', type=float, default=1.,
                        help='Weight to assign adverbs not in weighted vocabulary')
    parser.add_argument('-M', '--model', type=str, dest='model',
                        help='Model file for predicting adverb weights')
    args = parser.parse_args()
    
    
    # read pairs
    pairs = read_pairs(args.datadir)
    namelookup = {'rblr': rb_lr,
                  'socal': socal,
                  'patterns': patterns}
    
    # generate scores
    if args.method == 'patterns':
        scores = scale_scores(patterns.score(pairs))
    elif args.method == 'rblr':
        scores = scale_scores(rb_lr.score(pairs, args.model))
    elif args.method == 'socal':
        scores = scale_scores(socal.score(pairs))
    elif args.method.startswith('combined2'):
        __, m1name, m2name = args.method.split('_')
        m1_scores = scale_scores(namelookup[m1name].score(pairs))
        m2_scores = scale_scores(namelookup[m2name].score(pairs))
        m1_mean, m1_std = get_log_mean_std(m1_scores)
        m2_mean, m2_std = get_log_mean_std(m2_scores)
        scores = combine_scores(m1_scores, m2_scores, m1_mean, m1_std, m2_mean, m2_std,)
    elif args.method.startswith('combined3'):
        __, m1name, m2name, m3name = args.method.split('_')
        namelookup = {'pp': rb_lr,
                      'socal': socal,
                      'pat': patterns}
        m1_scores = scale_scores(namelookup[m1name].score(pairs))
        m2_scores = scale_scores(namelookup[m2name].score(pairs))
        m3_scores = scale_scores(namelookup[m3name].score(pairs))
        m1_mean, m1_std = get_log_mean_std(m1_scores)
        m2_mean, m2_std = get_log_mean_std(m2_scores)
        m3_mean, m3_std = get_log_mean_std(m3_scores)
        scores = combine_three_scores(m1_scores, m2_scores, m3_scores, m1_mean, m1_std, m2_mean, m2_std, m3_mean, m3_std)        
    
    # print scores
    if args.outfile is None:
        fout = sys.stdout
    else:
        fout = open(args.outfile, 'w')
    print_scores(scores, fout)