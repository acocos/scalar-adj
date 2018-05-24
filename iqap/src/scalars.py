#!/usr/bin/env python

from collections import defaultdict
import csv
from glob import glob
import numpy as np
from operator import itemgetter
import re
import yaml

sentiment = defaultdict(int)
for sentiment_filename in glob("wn/*.yaml"):
    d = yaml.load(open(sentiment_filename, 'rt'))
    for key,val in list(d.items()):
        sentiment[key] = val

class Evaluator:
    def __init__(self, dictionary_filename, dialogues_filename, predictions_filename):
        self.dictionary = Dictionary(dictionary_filename)
        self.dialogues_filename = dialogues_filename
        self.predictions_filename = predictions_filename
        self.annotations = AnnotatedDialogues(dialogues_filename)


    def with_means(self):
        predictions = {}
        for dialogue in self.annotations.dialogues:
            prediction = self.decision(
                dialogue.modsQ,
                dialogue.modsA,
                dialogue.negation,
                "meanfreq",
                dialogue.classification)
            predictions[dialogue.hitid] = [dialogue.tri_dominant_answer, prediction]
        return predictions

    def with_maxs(self):
        predictions = {}
        for dialogue in self.annotations.dialogues:
            prediction = self.decision(
                dialogue.modsQ,
                dialogue.modsA,
                dialogue.negation,
                "maxfreq",
                dialogue.classification)
            predictions[dialogue.hitid] = [dialogue.tri_dominant_answer, prediction]
        return predictions

    def with_wordnet(self):
         predictions = {}
         for dialogue in self.annotations.dialogues:             
             prediction = self.decision(
                 dialogue.modsQ,
                 dialogue.modsA,
                 dialogue.negation,
                 "sentiment",
                 dialogue.classification)
             predictions[dialogue.hitid] = [dialogue.tri_dominant_answer, prediction]
         return predictions

    def decision(self, modsQ, modsA, negation, funcname, classification):
        if classification == "avoided_adjective.txt":
            modQstr = modsQ[0].split(" ")[0]
            modQ = self.dictionary.phrase(modQstr)
            if modQ == None:
                return "uncertain"
            else:            
                val = eval("modQ." + funcname)            
                if val < 0:
                    return "no"
                else:
                    return "yes"
        else:        
            for modQstr in modsQ:
                for modAstr in modsA:
                    modQ = self.dictionary.phrase(modQstr)
                    modA = self.dictionary.phrase(modAstr)
                    if modQ == None or modA == None:
                        pass # Skip this pair.
                    elif modQ.name == modA.name:
                        return self.__reverse_prediction("yes", negation)
                    elif np.sign(eval("modQ." + funcname)) != np.sign(eval("modA." + funcname)):
                        return self.__reverse_prediction("no", negation)
                    elif abs(eval("modQ." + funcname)) <= abs(eval("modA." + funcname)):
                        return self.__reverse_prediction("yes", negation)
                    elif abs(eval("modQ." + funcname)) >= abs(eval("modA." + funcname)):
                        return self.__reverse_prediction("no", negation)
            return "uncertain"

    def __reverse_prediction(self, prediction, negation):        
        if prediction == "yes" and negation != "":
            return "no"
        elif prediction == "no" and negation != "":
            return "yes"
        else:
            return prediction
                    
    def create_predictions_file(self, *predictions_sets):
        with open(self.predictions_filename, "wt") as f:
            csv_writer = csv.writer(f, skipinitialspace=True, quotechar='"', delimiter=',')
            # Header,
            header = self.annotations.fieldnames
            for predictions_set in predictions_sets:            
                method = predictions_set[1]
                header.append("Prediction_" + method)
                header.append(method + "_IsAccurate")
            csv_writer.writerow(header)
            # Predictions.        
            for dialogue in self.annotations.dialogues:
                new_row = dialogue.row
                for predictions_set in predictions_sets:
                    predictions = predictions_set[0]
                    is_accurate = 0
                    actual, predicted = predictions[dialogue.hitid]
                    if actual == predicted:
                        is_accurate = 1
                    new_row += [predicted, is_accurate]                    
                csv_writer.writerow(new_row)                
        # Print the correct/incorrect distribution.        
        for predictions_set in predictions_sets:
            counts = defaultdict(int)
            predictions, method = predictions_set
            for hitid, vals in list(predictions.items()):
                if vals[0] == vals[1]:
                    counts["CORRECT"] += 1
                else:
                    counts["INCORRECT"] += 1
            print("\n%s" % method)
            for verdict, count in list(counts.items()):
                per = float(count)/float(sum(counts.values()))
                print("%s\t%s (%s)" % (verdict,count,per))

            
######################################################################

class AnnotatedDialogues:
    def __init__(self, filename):
        self.filename = filename
        self.rows = list(csv.reader(open(filename), delimiter=',', quotechar='"'))
        self.fieldnames = self.rows[0]
        self.rows.pop(0)
        self.dialogues = list(map((lambda x : Dialogue(x, self.fieldnames)), self.rows))
    
class Dialogue:
    def __init__(self, row, fieldnames):
        self.fieldnames = fieldnames
        self.row = row
        for i in range(0, len(self.fieldnames)):            
            setattr(self, self.__convert_fieldname(self.fieldnames[i]), self.row[i])
        self.modsQ = list(map((lambda x : x.strip().lower()), self.adjective_a.split("/")))
        self.modsA = []
        for adj in map((lambda x : x.strip().lower()), self.adjective_b.split("/")):
            self.modsA.append(adj)
            if self.adverb:
                self.modsA.append(self.adverb + " " + adj)            

    def __convert_fieldname(self, fieldname):
        fieldname = fieldname.replace(".", "_")
        fieldname = re.sub(r"([a-z])([A-Z])", r"\1_\2", fieldname)
        fieldname = fieldname.lower()
        return fieldname

######################################################################

class Dictionary:
    def __init__(self, filenames):
        self.filenames = filenames
        phrases2rows = defaultdict(list)
        rows = []
        for filename in filenames:
            these_rows = list(csv.reader(open(filename), delimiter=',', quotechar='"'))
            these_rows.pop(0)
            rows += these_rows
        for row in rows:
            phrases2rows[row[0]].append(row)
        self.phrases = []
        for row_set in list(phrases2rows.values()):
            self.phrases.append(Phrase(row_set))

    def phrase(self, phrasename):
        for phrase in self.phrases:
            if phrase.name == phrasename:
                return phrase
        return None #phrasename, "not found in", self.filename

class Phrase:
    def __init__(self, rows):
        self.rows = rows
        self.name = rows[0][0]
        self.freqs = self.get_freqs()
        self.maxfreq = self.get_max_freq()
        self.meanfreq = self.get_mean_freq()
        self.sentiment = sentiment[self.name]

    def get_freqs(self):
        f = {}
        for row in self.rows:
            phrase,rating,freq,total = row
            f[float(rating)-3] = float(freq)/float(total)
        dist = {}
        for rating,freq in list(f.items()):
            if sum(f.values()) > 0:
                dist[rating] = freq/sum(f.values())
            else:
                dist[rating] = 0
        return dist

    def get_max_freq(self):
        return sorted(list(self.freqs.items()), key=itemgetter(1), reverse=True)[0][0]

    def get_mean_freq(self):
        mu = 0.0
        for rating,freq in list(self.freqs.items()):
            mu += float(rating) * float(freq)
        return mu

        
        
