#!/usr/bin/env python

'''
score.py

Score results of IQAP experiments.

Since our scalar_pairs.py code does not implement the combined 
metrics on its own, we implement them in the scoring by assuming
metrics have a coefficient of 0 if the prediction is 'uncertain'
'''
import os, sys
import csv
import re
import pickle as pkl
from collections import Counter

from sklearn.metrics import precision_recall_fscore_support
from scalars import AnnotatedDialogues, Dialogue

        
def prf_macro(y, yhat):
    yclean, yhatclean = zip(*[(yy,yh) for yy,yh in zip(y,yhat) if yy!='uncertain'])
    classes = set(yclean)
    ps = []
    rs = []
    fs = []
    for cls in classes:
        p,r,f = prf(yclean, yhatclean, cls)
        ps.append(p)
        rs.append(r)
    n = len(classes)
    pavg = sum(ps)/n
    ravg = sum(rs)/n
    favg = (2*pavg*ravg)/(pavg+ravg)
    return pavg, ravg, favg

def prf(y, yhat, cls):
    ymod = [yy==cls for yy in y]
    yhatmod = [yy==cls for yy in yhat]
    correct = [yy*yh for yy,yh in zip(ymod, yhatmod)]
    try:
        p = float(sum(correct))/sum(yhatmod)
    except ZeroDivisionError:
        p = 0.
    try:
        r = float(sum(correct))/sum(ymod)
    except ZeroDivisionError:
        r = 0.
    try:
        f = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f = 0.
    return p,r,f

def acc(y, yhat):
    corr = [float(yy==yh) for yy,yh in zip(y,yhat)]
    return sum(corr)/len(corr)

if __name__=="__main__":
    
    origdatafile = sys.argv[1] # ../data/indirect-answers.combined.imdb-predictions.csv
    origdata = AnnotatedDialogues(origdatafile)
    
    resultsdir = sys.argv[2]
    
    # read original data
    data = {d.hitid: {'hitid': d.hitid,
                      'classification': d.classification,
                      'negation': d.negation,
                      'adverb': d.adverb,
                      'modsQ': d.modsQ,
                      'modsA': d.modsA,
                      'actual': d.tri_dominant_answer,
                      'p_demarneffe': d.prediction_means,
                      'corr_demarneffe': d.prediction_means==d.tri_dominant_answer} 
            for d in origdata.dialogues}
    
    # read predictions
    for method in ['rblr', 'socal','patterns']:
        rdr = csv.DictReader(open(os.path.join(resultsdir, '%s_results.elasticnet.csv'%method), 'r'))
        for row in rdr:
            data[row['hitid']]['p_%s'%method] = row['prediction']
            data[row['hitid']]['corr_%s'%method] = row['prediction']==row['actual']
    
    # Generate combined 2-metrics
    combos = [['patterns','socal'],
              ['patterns','rblr'],
              ['rblr','socal'],
              ['rblr','patterns'],
              ['socal','patterns'],
              ['socal','rblr']]
    for m1, m2 in combos:
        for hid, hdct in data.items():
            s_m1 = hdct['p_%s' % m1]
            s_m2 = hdct['p_%s' % m2]
            act = hdct['actual']
            if s_m1 != 'uncertain':
                comb = s_m1
            else:
                comb = s_m2
            hdct['p_%s+%s' % (m1, m2)] = comb
            hdct['corr_%s+%s' % (m1, m2)] = comb==act
    
    # Generate combined 3-metrics
    combos = [['patterns','socal','rblr'],
              ['patterns','rblr','socal'],
              ['rblr','socal','patterns'],
              ['rblr','patterns','socal'],
              ['socal','patterns','rblr'],
              ['socal','rblr','patterns']]
    for m1, m2, m3 in combos:
        for hid, hdct in data.items():
            s_m1 = hdct['p_%s' % m1]
            s_m2 = hdct['p_%s' % m2]
            s_m3 = hdct['p_%s' % m3]
            act = hdct['actual']
            if s_m1 != 'uncertain':
                comb = s_m1
            elif s_m2 != 'uncertain':
                comb = s_m2
            else:
                comb = s_m3
            hdct['p_%s+%s+%s' % (m1, m2, m3)] = comb
            hdct['corr_%s+%s+%s' % (m1, m2, m3)] = comb==act
    
    # sort by question class
    classifications = set([r['classification'].split('_')[0] for r in data.values()])
    data_byclass = {c: [d for hid, d in data.items() if d['classification'].split('_')[0]==c] for c in classifications}
    
    cls = 'adjectives' # we only care about this one for now
    dctlist = data_byclass[cls]
    
    y = [d['actual'] for d in dctlist]
    
    # write results to CSV
    methods = [k.replace('corr_','') for k in data['78'].keys() if k.startswith('corr_')]
    print "metric,oov,acc,p,r,f"
    for method in methods:
        yhat = [d['p_%s'%method] for d in dctlist]
        yclean, yhatclean = zip(*[(yy,yh) for yy,yh in zip(y,yhat) if yy!='uncertain'])
        n = float(len(yhatclean))
        oov = Counter(yhatclean).get('uncertain',0.)/n
        p,r,f = prf_macro(yclean, yhatclean)
        accuracy = acc(yclean,yhatclean)
        print method+','+",".join(['%0.3f' % v for v in [oov, accuracy, p, r, f]])
    #all-yes baseline
    method="allYES"
    yhat = ['yes' for d in dctlist]
    yclean, yhatclean = zip(*[(yy,yh) for yy,yh in zip(y,yhat) if yy!='uncertain'])
    n = float(len(yhatclean))
    oov = 0.
    p,r,f = prf_macro(yclean, yhatclean)
    accuracy = acc(yclean,yhatclean)
    print method+','+",".join(['%0.3f' % v for v in [oov, accuracy, p, r, f]])
