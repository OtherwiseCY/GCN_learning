import os
import numpy as np

def _parse_cora_features_labels(path):

    id2index = {}

    label2index = {
        'Case_Based': 0,
        'Genetic_Algorithms': 1,
        'Neural_Networks': 2,
        'Probabilistic_Methods': 3,
        'Reinforcement_Learning': 4,
        'Rule_Learning': 5,
        'Theory': 6
    }

    features = []
    labels = []

    i = 0
    with open(path + 'cora.content', 'r') as f:
        for line in f:
            items = line.strip().split('\t')

            id = items[0]

            labels.append([label2index[items[-1]]])

            # parse features
            features.append([int(x) for x in items[1:-1]])

            id2index[id] = i
            i += 1

    features = np.asarray(features, dtype='float32')
    labels = np.asarray(labels, dtype='int32')

    return features, labels, id2index

def data_parser():
    path = "../data/"
    features,labels,id2index = _parse_cora_features_labels(path)
    
    
    n_papers = len(id2index)

    adj = np.zeros((n_papers, n_papers), dtype='float32')

    with open(path + 'cora.cites', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            adj[id2index[items[0]], id2index[items[1]]] = 1.0
            adj[id2index[items[1]], id2index[items[0]]] = 1.0

    return adj.astype('float32'), features.astype('float32'), labels.astype('int32')
