import os, sys

SOCALFILE = '../data/adj_dictionary1.11.txt'

def read_data(f):
    d = {}
    with open(f,'rU') as fin:
        for line in fin:
            w, s = line.strip().split('\t')
            d[w] = int(s)
    return d


def flatten(l):
    return [item for sublist in l for item in sublist]

def score(pairs):
    vocab = set(flatten(pairs))
    socal = read_data(SOCALFILE)
    scores = {}
    for w in vocab:
        scores[w] = socal.get(w, None)
    return scores