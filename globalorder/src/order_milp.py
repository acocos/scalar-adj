#!/usr/bin/env python

'''
order_milp.py

Implementation of MILP for ordering scalar adjective
from DeMelo & Bansal 2013 (follows DeMelo code RankerILP.java)
'''

import os, sys
import numpy as np
import argparse
import itertools
from ortools.linear_solver import pywraplp

def load_terms(tf):
    terms = set()
    with open(tf, 'rU') as fin:
        for line in fin:
            __, t = line.strip().split('\t')
            terms.add(t)
    return terms

def load_weights(wf):
    weights = {}
    with open(wf, 'rU') as fin:
        for line in fin:
            w1, w2, wgt = line.strip().split('\t')
            weights[(w1,w2)] = float(wgt)
    return weights

def print_result(rankdct, fout):
    wgts = set(rankdct.values())
    wgtdct = {w: [] for w in wgts}
    for k,v in rankdct.items():
        wgtdct[v].append(k)
    for k in sorted(wgtdct.keys()):
        print >> fout, '\t'.join((str(k), ' || '.join(wgtdct[k])))

def scale_factor(wgtdct):
    scaling_factor = sys.maxint
    for v in wgtdct.values():
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
    return {k: sf*v for k,v in wgtdct.items()}


def rank_terms(terms, wgts, equivterms=set()):
    '''
    Implementation of term ranking Mixed Integer Linear Program
    by DeMelo and Bansal 2013
    :param: terms: set of string terms
    :param: wgts: dictionary of pairwise weights, with float values, e.g. {(term1, term2): weight}
    '''
    RANGE = 1.0
    DMIN = 0.0
    C = 1.0 + RANGE * 10
    if len(equivterms) > 0:
        synonymCoefficient = 10000000000
    else:
        synonymCoefficient = 0.000001
    
    # re-scale scores
    wordPairWeights = scale_scores(wgts)
    
    # start creating LP
    solver = pywraplp.Solver('SolveIntegerProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    ## assign indices to adjectives
    terms = sorted(list(terms))
    adjPosIdx = {}
    variables = []
    constraints = []
    objective = solver.Objective()
    for t in terms:
        idx = len(adjPosIdx)
        adjPosIdx[t] = idx
        variables.append(solver.NumVar(0.0, RANGE, 'x_%s' % t))
        objective.SetCoefficient(variables[idx], 0.0)
    
    ## set up constraints and coefficients
    idxOfs = len(adjPosIdx)
    deltaIdx = {}
    for adj1 in terms:
        adj1Idx = adjPosIdx[adj1]
        for adj2 in terms:
            if adj1 < adj2:
                ## for each (unique) pair of adjectives, we have one delta (!!!!)
                dijIdx = len(variables)
                adjPair = (adj1, adj2)
                deltaIdx[adjPair] = dijIdx
                adjPairWeight = wordPairWeights.get(adjPair, 0.0)
                sys.stderr.write("\nx%d : (%s,%s)\t (%0.4f)" % (dijIdx, adjPair[0], adjPair[1], adjPairWeight))
                variables.append(solver.NumVar(-RANGE, RANGE, 'd_{%s,%s}' % adjPair))
                
                ## enforce d_ij = x_j - x_i
                adj2Idx = adjPosIdx[adj2]
                constraints.append(solver.Constraint(0.0, 0.0))
                constraints[-1].SetCoefficient(variables[dijIdx], 1.0)
                constraints[-1].SetCoefficient(variables[adj2Idx], -1.0)
                constraints[-1].SetCoefficient(variables[adj1Idx], 1.0)
                
                sidx = len(variables)
                variables.append(solver.IntVar(0, 1, 's_{%s,%s}' % adjPair))
                widx = len(variables)
                variables.append(solver.IntVar(0, 1, 'w_{%s,%s}' % adjPair))
                
                epsilon = 0.0001
                # d_ij + s_ij * C >= 0
                constraints.append(solver.Constraint(DMIN, solver.infinity()))
                constraints[-1].SetCoefficient(variables[dijIdx], 1.0)
                constraints[-1].SetCoefficient(variables[sidx], C)
                
                # d_ij - (1 - s_ij)C < 0
                constraints.append(solver.Constraint(-solver.infinity(), DMIN + C - epsilon))
                constraints[-1].SetCoefficient(variables[dijIdx], 1.0)
                constraints[-1].SetCoefficient(variables[sidx], C)
                
                # d_ij - w_ij * C <= 0
                if DMIN > 0.0:
                    constraints.append(solver.Constraint(-solver.infinity(), -DMIN))
                else:
                    constraints.append(solver.Constraint(-solver.infinity(), 0.0))
                constraints[-1].SetCoefficient(variables[dijIdx], 1.0)
                constraints[-1].SetCoefficient(variables[widx], -C)
                
                # d_ij + (1 - w_ij)C > 0
                constraints.append(solver.Constraint(-DMIN-C+epsilon, solver.infinity()))
                constraints[-1].SetCoefficient(variables[dijIdx], 1.0)
                constraints[-1].SetCoefficient(variables[widx], -C)
                
                synonymPair = ((adj1, adj2) in equivterms)
                if synonymPair:
                    objective.SetCoefficient(variables[sidx], -synonymCoefficient)
                    objective.SetCoefficient(variables[widx], -synonymCoefficient)
                else:
                    objective.SetCoefficient(variables[sidx], -adjPairWeight)
                    objective.SetCoefficient(variables[widx], adjPairWeight)
                sys.stderr.write('\n')
    
    # run optimization
    objective.SetMaximization()
    result_status = solver.Solve()
    
    # get results
    objval = solver.Objective().Value()
    term_intensities = {t: variables[adjPosIdx[t]].solution_value() for t in terms}
    
    return (term_intensities, objval)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pairwise adjective ranking.')
    parser.add_argument('-t', '--termsfile', type=str, dest='termsfile',
                        help='File containing terms files for ranking')
    parser.add_argument('-w', '--weightsfile', type=str, dest='weightsfile',
                        help='File containing pairwise weights')
    parser.add_argument('-o', '--outfile', type=str, default=None)
    args = parser.parse_args()
    
    # read terms
    terms = load_terms(args.termsfile)
    
    # read pairwise weights
    weights = load_weights(args.weightsfile)
    
    # rank terms
    rankings, __ = rank_terms(terms, weights)
    
    # print results
    if args.outfile is None:
        fout = sys.stdout
    else:
        fout = open(args.outfile, 'w')
    print_result(rankings, fout)