#4i!/bin/python

# ignore annoying sklearn DeprecationWarning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import sys
import pdb
import random
import pickle
import math
import operator
import optparse
import itertools
import numpy as np
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from scipy.sparse import coo_matrix

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#read data and features from file
def get_data(label_file, feature_file, conf_file=None, le=None, dv=None, ss=None, encodeY=True, standardize=False) : 
    sys.stderr.write('Reading data...\n')
    nm = []; y = []; X = [];
    for i,(labels, feats) in enumerate(zip(open(label_file).readlines(), open(feature_file).readlines())) : 
        w1, w2, l = labels.strip().split('\t')
        x = {}
        if i % 10000 == 0 : sys.stderr.write('%s'%labels)
        for fv in feats.strip().split('\t') : 
            if len(feats.strip()) == 0: continue
            try : 
                f, v = fv.rsplit(':',1)
            except ValueError : 
                continue 
            try : 
                x[f.encode('ascii', 'ignore')] = float(v)
            except : 
                continue
        nm.append((w1,w2)) #,float(c)))
        y.append(l)
        X.append(x)
    sys.stderr.write('Encoding labels...\n')
    if le is None : 
        le = LabelEncoder()
        le.fit(list(set(y)))
    if encodeY : y = le.transform(y)
    sys.stderr.write('Transforming data...\n')
    if dv is None : 
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(X)
    else : X = dv.transform(X)
    return y, X, le, dv, nm

#train a logistic regression classifier
def train_classifier(X, Y,C=1,le=None, penalty='l2', l1_ratio=0.15):
    clf = LogisticRegression(C=C, class_weight='balanced', penalty=penalty)
    clf.fit(X,Y)
    return clf

#test the classifier
def test_classifier(clf, X, Y):
    return clf.score(X,Y)

def confusion(clf, X, Y, le) : 
    conf = {}
    names = set()
    for x,a in zip(X,Y) : 
        a = a[:4]
        p = le.inverse_transform([clf.predict(x)[0]])
        p = p[:4]
        if p not in conf : conf[p] = {}
        if a not in conf[p] : conf[p][a] = 0
        conf[p][a] += 1
        names.add(a)
    names = sorted(list(names))
    print '\t%s'%('\t'.join(names))
    for a in conf: 	
        row = [(0 if p not in conf[a] else conf[a][p]) for p in names]
        print '%s\t%s'%(a, '\t'.join([str(c) for c in row]))

def predict_labels(clf, X, nm, le, y_test=None) : 
    pred = []
    pred_probs = clf.predict_proba(X)
    for pr, probs in zip(nm, pred_probs):
        problist = list(probs)
        mx = max(problist)
        mxidx = problist.index(mx) 
        mxlbl = le.inverse_transform([mxidx])
        outline = '\t'.join(pr)
        outline += '\t%s' % mxlbl
        for i, p in enumerate(problist):
            outline += '\tprob-%s=%0.5f' % (le.inverse_transform([i])[0], p)
        print outline
        pred.append(mxlbl)
    if y_test is not None:
        P, R, F, N = precision_recall_fscore_support(y_test, pred)
        macro = precision_recall_fscore_support(y_test, pred, average='macro')
        micro = precision_recall_fscore_support(y_test, pred, average='micro')
        wghted = precision_recall_fscore_support(y_test, pred, average='weighted')
        A = accuracy_score(y_test, pred)
        sys.stderr.write('Test Accuracy:\t%0.4f\n' % A)
        sys.stderr.write('Test Macro avg.:\t\t\t%.04f\t%.04f\t|\t%.04f\n'% macro[:3])
        sys.stderr.write('Test Micro avg.:\t\t\t%.04f\t%.04f\t|\t%.04f\n'% micro[:3])
        sys.stderr.write('Test Weighted avg.:\t\t\t%.04f\t%.04f\t|\t%.04f\n'% wghted[:3])
    return pred

def binary_predict_labels(clfs) : 
    labels = []
    classes = clfs.keys() 
    for rel in classes : 
        (clf, le, _, X) = clfs[rel]
        l = []
        for x in X : 
            prob = clf.predict_proba(x)[0][le.transform(["yes"])[0]]
            l.append(prob)
        labels.append(l)
    for i in range(X.shape[0]): 
        l = []; v = []
        for j, c in enumerate(classes) :  
            l.append('%s=%f'%(c,labels[j][i]))
            v.append(labels[j][i])
        mx = classes[v.index(max(v))]
        print '%s\t%s'%(mx,'\t'.join(['%s'%ll for ll in l]))

def binary_print_features(clfs) : 
    for rel in clfs : 
        (clf, le, dv, X) = clfs[rel]
        fnames = dv.get_feature_names()
        print rel
        sorted_weights = sorted([(fnames[i],w) for i,w in enumerate(clf.coef_[0])], key=lambda e: e[1], reverse=True)
        for i,w in sorted_weights[:20]:
            print '%s\t%f'%(i,w)

#precision, recall, F1
def prf(X_train, y_train, X_test, y_test, C=1.0, l1_ratio=0.15) : 
    clf = train_classifier(X_train, y_train, C=C, l1_ratio=l1_ratio)
    pred = clf.predict(X_test)
    pred_probs = clf.predict_proba(X_test)
    P, R, F, N = precision_recall_fscore_support(y_test, pred)
    macro = precision_recall_fscore_support(y_test, pred, average='macro')
    micro = precision_recall_fscore_support(y_test, pred, average='micro')
    wghtd = precision_recall_fscore_support(y_test, pred, average='weighted')
    A = test_classifier(clf, X_test, y_test)
    scores = []
    for i in range(len(P)) : 
        scores.append((A,P[i],R[i],F[i], macro[2], micro[2], wghtd[2]))
    cm = np.zeros((len(P), len(P))) 
    for t, p in zip(y_test, pred) : 
        cm[t][p] += 1.0 / len(y_test)
    return scores, cm, clf, pred_probs

#normal cross validation	
def cross_validate(X, Y, nm, sX, sy, le, dv, numfolds=10, printing=True, c=1.0, scoresfile=None) : 
    num_classes = len(set(Y))
    train_accs = [[] for _ in range(num_classes)]
    test_accs = [[] for _ in range(num_classes)]
    cm = np.zeros((num_classes, num_classes)) 
    nm = np.array(nm)
    class_totals = None
    skf = StratifiedKFold(n_splits=numfolds)
    for n, (train_idx, test_idx) in enumerate(skf.split(X,Y)):
        x_train = X[train_idx]
        y_train = Y[train_idx]
        nm_train = nm[train_idx]
        x_test = X[test_idx]
        y_test = Y[test_idx]
        nm_test = nm[test_idx]
        train_prf, train_conf, __, __ = prf(x_train, y_train, x_train, y_train, C=c)
        test_prf, test_conf, clf, test_probs = prf(x_train, y_train, x_test, y_test, C=c)
        for t in range(test_conf.shape[0]) : cm[t] += test_conf[t]
        if class_totals is None : class_totals = np.bincount(y_test.astype(int))
        else : class_totals += np.bincount(y_test.astype(int))
        for i, scores in enumerate(train_prf) : train_accs[i].append(np.array(scores))
        for i, scores in enumerate(test_prf) : test_accs[i].append(np.array(scores))
        for i, scores in enumerate(test_prf) : 
            a, p, r, f, macro, micro, wghtd  = scores
            if printing : print 'Fold %d (%s):\t%.04f\t%.04f\t%.04f\t|%.04f'%(n,le.inverse_transform([i])[:5],a,p,r,f)
        if printing : print
        if scoresfile is not None:
            with open(scoresfile, 'a') as fout:
                for pr, probs in zip(nm_test, test_probs):
                    problist = list(probs)
                    mx = max(problist)
                    mxidx = problist.index(mx)
                    mxlbl = le.inverse_transform([mxidx])
                    outline = '\t'.join(pr)
                    outline += '\t%s' % mxlbl
                    for i, p in enumerate(problist):
                        outline += '\tprob-%s=%0.5f' % (le.inverse_transform([i])[0], p)
                    print >> fout, outline
    num_classes = 0
    for i,t in enumerate([sum(t)/numfolds for t in test_accs]) : 
        a, p, r, f, macro, micro, wghtd = tuple(t)
        sys.stderr.write('Test average (%s):\t%.04f\t%.04f\t%.04f\t|\t%.04f\n'%(le.inverse_transform([i])[:5],a,p,r,f))
        print 'Test average (%s):\t%.04f\t%.04f\t%.04f\t|\t%.04f'%(le.inverse_transform([i])[:5],a,p,r,f)
    sys.stderr.write('\t\t\t\t\t\t|\t------\n')
    print '\t\t\t\t\t\t|\t------'
    sys.stderr.write('Macro avg.:\t\t\t\t\t|\t%.04f\n'%(macro))
    print 'Macro avg.:\t\t\t\t\t|\t%.04f'%(macro)
    sys.stderr.write('Micro avg.:\t\t\t\t\t|\t%.04f\n'%(micro))
    print 'Micro avg.:\t\t\t\t\t|\t%.04f'%(micro)
    sys.stderr.write('Weighted avg.:\t\t\t\t\t|\t%.04f\n'%(wghtd))
    print 'Weighted avg.:\t\t\t\t\t|\t%.04f'%(wghtd)
    return clf
    if printing : 
        np.set_printoptions(suppress=True, precision=3)
        avg_class_count = class_totals / numfolds
        print avg_class_count
        print avg_class_count.astype(float) / sum(avg_class_count)
        print cm / numfolds


def flatten(l):
    return [item for sublist in l for item in sublist]


def print_features(clf, dv, le) :
    N = 100000
    weights = clf.coef_
    fnames = dv.feature_names_
    for i, ws in enumerate(weights) : 
        print le.inverse_transform([i])
        printed = 0
        for idx,w in sorted(enumerate(ws), key=lambda e: e[1], reverse=True) : 
            f = fnames[idx]
            if printed < N : 
                print '%s\t%.04f'%(f, w)
                printed += 1
        print

def unison_shuffled_copies(a, b, c, seed=0):
    sys.stderr.write('Shuffling...\n')
    assert a.shape[0] == b.shape[0] == c.shape[0]
    np.random.seed(seed)
    p = np.random.permutation(a.shape[0])
    return a[p], b[p], c[p]

if __name__ == '__main__' : 

    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--dir", dest="dir",  default="data/", help="Root data directory")
    optparser.add_option("-f", "--features", dest="features",  default="features.txt", help="File containing extracted features")
    optparser.add_option("-c", "--c", dest="c",  default=1.0, type="float", help="Regularization constant")
    optparser.add_option("-F", "--foldsize", dest="foldsize",  default=2000, type="int", help="Number of samples to test in each fold of cross validation")
    optparser.add_option("-n", "--n_fold", dest="n_fold",  default=10, type="int", help="Number of folds in cross validation")
    optparser.add_option("-t", "--tune", dest="tune",  default=False, action="store_true", help="Tune regularization parameter")
    optparser.add_option("-m", "--model", dest="model", help="File to read model from/write model to.")
    optparser.add_option("-v", "--save", dest="save",  default=False, action="store_true", help="Train a model and save it to the specified file.")
    optparser.add_option("-p", "--predict", dest="predict",  default=False, action="store_true", help="Train a model and save it to the specified file.")
    optparser.add_option("-s", "--standardize", dest="standardize", default=False, action="store_true", help="Standardize all features to have 0 mean, 1 variance")
    optparser.add_option("-i", "--id", dest="id", default='', type="str", help="ID for this experiment")
    optparser.add_option("-l", "--l1ratio", dest="l1ratio", default=0.15, type=float, help="L1 ratio for elastic net")
    optparser.add_option("--print_features", dest="print_features",  default=False, action="store_true", help="Print classifier's top features.")
    optparser.add_option("--conf", dest="conf",  default=0.0 , type="float", help="Only train on examples which have confidence at least conf.")

    (opts, _) = optparser.parse_args()

    label_file = "%s/labels"%opts.dir
    conf_file = "%s/confidences"%opts.dir
    feature_file = "%s/%s"%(opts.dir, opts.features)

    np.random.seed(40)
    if opts.id == '':
        split_seed = 0
    else:
        split_seed = ord(opts.id[0])

    if opts.predict or opts.print_features : 
        bundle = pickle.load(open(opts.model))
        clf = bundle['clf']
        le = bundle['le']
        dv = bundle['dv']
        try:
            ss = bundle['ss']
        except KeyError:
            ss = None
        if opts.predict:
            y, X, _, _, nm = get_data(label_file, feature_file, conf_file, le, dv, ss, encodeY=False, standardize=opts.standardize)
    else : y, X, le, dv, nm = get_data(label_file, feature_file, conf_file, standardize=opts.standardize)
    sy = None; sX = None;
    ss = None
    if opts.tune: 
        for c in [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000] : 
            for l1r in [0.15]:
                X, y, nm = unison_shuffled_copies(X, y, np.array(nm), seed=split_seed)
                print "\nC=%f  L1Ratio=%f" % (c, l1r)
                sys.stderr.write("C=%f  L1Ratio=%f\n" % (c, l1r))
                clf = cross_validate(X, y, nm, sX, sy, le, dv, numfolds=opts.n_fold, printing=True, c=c, scoresfile="%s/predictions/predictions.crossval.%s.c%0.4f.l1r%0.2f" % (opts.dir, opts.id, c, l1r))
    else : 
        if opts.print_features : 
            print_features(clf, dv, le)
        elif opts.save: 
            clf = train_classifier(X, y, C=opts.c, le=le, l1_ratio=opts.l1ratio)
            print "Training accuracy: ", test_classifier(clf, X, y)
            bundle = {'clf' : clf, 'le' : le, 'dv' : dv, 'ss': ss}
            pickle.dump(bundle, open(opts.model, 'w'))
        elif opts.predict:
            lbls = predict_labels(clf, X, nm, le, y_test=y)
        else : 
            X, y, nm = unison_shuffled_copies(X, y, np.array(nm), seed=split_seed)
            cross_validate(X, Y, nm, sX, sy, le, dv, numfolds=opts.n_fold, printing=True, c=c, scoresfile="%s/predictions/predictions.crossval.%s.c%0.4f.l1r%0.2f" % (opts.dir, opts.id, c, l1r))
