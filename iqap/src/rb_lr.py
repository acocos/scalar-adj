#!/usr/bin/env python

'''
rb_lr.py

Score pairwise adjective intensity based on logistic regression classifier trained 
on RB+JJ-->JJ paraphrase patterns
'''

import os, sys
import gzip
import json
import pickle
from networkx.readwrite import json_graph
from networkx import shortest_path
from networkx import NetworkXNoPath

import classify

GRAPHFILE = '../../jjgraph/jjgraph.json'
RBVOCAB = '../../pairpredict/data/rbmin10.txt'
DEFAULTMODEL = '../../pairpredict/data/trainval_jjgraph/models/model.c0.0001.adverb_binary.elasticnet.nobias.trainval_jjgraph.pkl'

def flatten(l):
    return [item for sublist in l for item in sublist]

def graph_from_json(f):
    with open(f, 'r') as infile:
        networkx_graph = json_graph.node_link_graph(json.load(infile))
    return networkx_graph

def load_weights(fname, baseline=False):
    d = {}
    if baseline:
        return d
    with open(fname, 'rU') as fin:
        for line in fin:
            rb, wgt = line.strip().split('\t')
            d[rb] = float(wgt)
    return d


def score(pairs, modelfile=DEFAULTMODEL):
    rb_vocab = set([l.split()[0] for l in open(RBVOCAB,'rU').readlines()])
    return score_nopaths(pairs, modelfile, rb_vocab)


def score_nopaths(pairs, modelfile, rb_vocab):
    '''
    
    '''
    # read pairs
    orderedpairs = []
    for p1, p2 in pairs:
        orderedpairs.extend([(p1,p2)])
    
    # extract edges we need to predict (direct only in this case)
    orderedfeats = []
    G = graph_from_json(GRAPHFILE)
    for x,y in orderedpairs : 
        feats = {}
        if x in G:
            edges = G[x].get(y, {})
            for edgeid, edgedct in edges.items():
                rb = edgedct.get('adverb', None)
                if rb is not None:
                    if rb_vocab is not None and rb not in rb_vocab:
                        continue
                    feats['edge-xy-rb-%s' % rb] = 1.
        if y in G:
            edges = G[y].get(x, {})
            for edgeid, edgedct in edges.items():
                rb = edgedct.get('adverb', None)
                if rb is not None:
                    if rb_vocab is not None and rb not in rb_vocab:
                        continue
                    feats['edge-yx-rb-%s' % rb] = 1.
        orderedfeats.append(('%s:%s'%(f,v) for f,v in feats.iteritems()))
    
    # make predictions
    label_file = 'templabels'
    feature_file = 'tempfeats'
    with open(label_file,'w') as fout1, open(feature_file,'w') as fout2:
        for pair, feats in zip(orderedpairs, orderedfeats):
            p1, p2 = pair
            print >> fout1, '\t'.join((p1, p2, '0'))
            print >> fout2, '\t'.join(feats)
    bundle = pickle.load(open(modelfile))
    clf = bundle['clf']
    le = bundle['le']
    dv = bundle['dv']
    try:
        ss = bundle['ss']
    except KeyError:
        pass
    try:
        y, X, _, _, nm, _ = classify.get_data(label_file, feature_file, le=le, dv=dv, ss=None, encodeY=False, standardize=False)
    except ValueError:
        y, X, _, _, nm = classify.get_data(label_file, feature_file, le=le, dv=dv, ss=None, encodeY=False, standardize=False)
    pred_probs = clf.predict_proba(X)
    predweights = {}

    pred_probs = clf.predict_proba(X)
    predweights = {}
    
    for i, (pair, probs) in enumerate(zip(orderedpairs, pred_probs)):
        if sum(X[i])==0:
            predweights[pair] = 0.5
        else:
            predweights[pair] = probs[1]
            
    
    # finalize scores
    scores = {}
    for x,y in pairs : 
        W_x = predweights[(x,y)]
        if W_x==0:
            scr = 0.
        else:
            scr = W_x - 0.5
        scores[(x,y)] = scr
    return scores
