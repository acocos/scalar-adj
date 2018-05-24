'''
Create training set from all edges in JJGraph
'''
import os, sys
import json
from networkx.readwrite import json_graph

GRAPHFILE = '../../jjgraph/jjgraph.json'

def graph_from_json(f):
    with open(f, 'r') as infile:
        networkx_graph = json_graph.node_link_graph(json.load(infile))
    return networkx_graph

def splitpop(s, delim='_'):
    ss = s.split(delim)
    return delim.join(ss[:-1]), ss[-1]

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

########################
# create JJGraph training set
########################

G = graph_from_json(GRAPHFILE)
jj_vocab = set(G.nodes())

for (j1, j2, d) in G.edges(data=True):
    if 'adverb' not in d: 
        continue
    print '\t'.join((j1, j2, '1'))
    print '\t'.join((j2, j1, '0'))
