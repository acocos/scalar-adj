import sys
import gzip
import json
from networkx.readwrite import json_graph


def graph_from_json(f):
    with open(f, 'r') as infile:
        networkx_graph = json_graph.node_link_graph(json.load(infile))
    return networkx_graph

GRAPHFILE = '../../jjgraph/jjgraph.json'

G = graph_from_json(GRAPHFILE)

if len(sys.argv) > 1:
    filtfile = sys.argv[1]
    rb_vocab = set([l.strip().split('\t')[0] for l in open(filtfile,'rU').readlines()])
else:
    rb_vocab = None

pairs = []
for line in sys.stdin : 
    x, y, l = line.strip().split('\t')
    pairs.append((x,y))


for x,y in pairs : 
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
  print '\t'.join(['%s:%s'%(f,v) for f,v in feats.iteritems()])
